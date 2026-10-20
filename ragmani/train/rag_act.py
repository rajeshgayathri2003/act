"""Trains RagAct on encoded demos with the Hugging Face Trainer, as rag_model.py does.

The loss and the truth schedule live in compute_loss, which reads training progress from the Trainer's own state.
Run from act/ as `python -m ragmani.train.rag_act <config> [section.key=value ...]`.
"""
import math
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from safetensors.torch import save_model
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from ragmani import data, simulate
from ragmani.config import load
from ragmani.data.retriever import Retriever
from ragmani.data.splits import SPLITS
from ragmani.models import RagAct

COLUMNS = ["image_emb", "state", "next_state", "action", "is_pad", "episode"]


def retrieval_metrics(idx, inputs, retriever):
  """How varied and how close the retrieved rows are: share of distinct rows, and mean similarity to the row's own frame."""
  return {
    "retrieval/distinct": len(idx.unique()) / idx.numel(),
    "retrieval/similarity": (retriever.rows["image_emb"][idx] * inputs["image_emb"][:, None]).sum(dim=-1).mean().item(),
  }


def collate(rows):
  return {key: torch.stack([row[key] for row in rows]) for key in COLUMNS}


class Best(TrainerCallback):
  """Keeps <run_dir>/best as the model at its lowest eval loss so far, with its step and epoch."""

  def __init__(self, run_dir):
    self.directory = run_dir / "best"
    self.lowest = math.inf

  def on_train_begin(self, args, state, control, **kwargs):
    if state.best_metric is not None: self.lowest = state.best_metric  # a resumed run's state carries its best eval loss, so best/ is not overwritten by a worse model

  def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
    if metrics["eval_loss"] >= self.lowest: return
    self.lowest = metrics["eval_loss"]
    self.directory.mkdir(parents=True, exist_ok=True)
    save_model(model, str(self.directory / "model.safetensors"))
    state.save_to_json(str(self.directory / "trainer_state.json"))


class RagActTrainer(Trainer):
  """Runs RagAct with the retriever; training gives each row its truth row with probability 1 - progress through the run."""

  def __init__(self, *args, retriever, pad_weight, exclude_own_episode, **kwargs):
    super().__init__(*args, **kwargs)
    self.retriever = retriever
    self.pad_weight = pad_weight
    self.exclude_own_episode = exclude_own_episode

  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    truth, add_truth = None, None
    if model.training:
      progress = self.state.global_step / max(self.state.max_steps, 1)  # 0 at the start of the run, 1 at the end
      truth = (inputs["action"], inputs["next_state"])
      add_truth = torch.rand(len(inputs["state"]), device=inputs["state"].device) < 1 - progress  # each row on its own draw

    exclude = inputs["episode"] if self.exclude_own_episode else None
    pred_action, pred_pad, idx, logits = model(inputs["image_emb"], inputs["state"], self.retriever, exclude=exclude, truth=truth, add_truth=add_truth)

    target_action = (inputs["action"] - model.action.mean) / model.action.std
    real_mask = (~inputs["is_pad"]).unsqueeze(-1).float()  # 1 on chunk steps before the demo's end
    real_values = (real_mask.sum() * pred_action.shape[-1]).clamp(min=1)
    error = F.l1_loss(pred_action, target_action, reduction="none") * real_mask
    losses = {
      "action_l1": error.sum() / real_values,  # mean over real steps only
      "pad_bce": F.binary_cross_entropy_with_logits(pred_pad, inputs["is_pad"].float()),
    }
    if logits is not None: losses["action_ce"] = (model.codebook.loss(logits, target_action) * real_mask).sum() / real_values

    if not model.training:
      self.eval_rows += len(inputs["state"])
      for name, value in losses.items(): self.eval_losses[name] += value.item() * len(inputs["state"])
    elif self.state.global_step % self.args.logging_steps == 0:
      wandb.log({"train/global_step": self.state.global_step, **{f"train/{name}": value.item() for name, value in losses.items()}, **retrieval_metrics(idx, inputs, self.retriever)})

    action_loss = "action_ce" if logits is not None and model.training else "action_l1"  # eval_loss stays L1, so best/ and runs with either head compare
    loss = losses[action_loss] + self.pad_weight * losses["pad_bce"]
    return (loss, {"action": pred_action, "is_pad": pred_pad}) if return_outputs else loss

  def evaluate(self, *args, **kwargs):
    """The Trainer's eval, plus the two parts of the eval loss logged on their own."""
    self.eval_losses, self.eval_rows = defaultdict(float), 0
    metrics = super().evaluate(*args, **kwargs)
    wandb.log({"train/global_step": self.state.global_step, **{f"eval/{name}": total / self.eval_rows for name, total in self.eval_losses.items()}})
    return metrics

  def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    """Eval loss through compute_loss; the default calls model(**inputs), which needs a labels column we do not have."""
    with torch.no_grad(): loss = self.compute_loss(model, self._prepare_inputs(inputs))
    return loss.detach(), None, None


def main(name, overrides):
  config = load(name, overrides)

  # train and eval rows
  dataset = data.load(config.paths.dataset).with_format("torch", columns=COLUMNS)
  is_held_out = SPLITS[config.train.split](dataset["episode"][:].numpy(), config.train.eval_fraction, config.train.split_seed)
  train_rows = dataset.select(np.flatnonzero(~is_held_out))
  eval_rows = dataset.select(np.flatnonzero(is_held_out))
  if not len(eval_rows): raise ValueError(f"train.eval_fraction={config.train.eval_fraction} holds out no rows of this dataset under the {config.train.split} split")

  # retrieval memory and model
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  retriever = Retriever(train_rows, config.retrieval, device)  # memory is training rows only
  model = RagAct(config.model, image_dim=train_rows[0]["image_emb"].shape[-1])

  if config.model.normalize:
    stats = data.statistics(train_rows)
    model.state.mean, model.state.std = stats["state"]["mean"], stats["state"]["std"]
    model.action.mean, model.action.std = stats["action"]["mean"], stats["action"]["std"]

  print(
    f"{config.train.split} split: {len(train_rows)} training rows, {len(eval_rows)} eval rows; "
    f"own episode excluded from retrieval: {config.train.exclude_own_episode}; "
    f"{sum(parameter.numel() for parameter in model.parameters())} parameters"
  )

  arguments = TrainingArguments(
    output_dir=str(config.paths.run_dir),
    num_train_epochs=config.train.epochs,
    per_device_train_batch_size=config.train.batch_size,
    per_device_eval_batch_size=config.train.batch_size,
    learning_rate=config.train.lr,
    lr_scheduler_type=config.train.lr_scheduler,
    lr_scheduler_kwargs=dict(config.train.plateau) if config.train.lr_scheduler == "reduce_lr_on_plateau" else {},
    warmup_steps=float(config.train.warmup_ratio),  # transformers v5 dropped warmup_ratio; a float in [0, 1) is read as a ratio
    max_grad_norm=config.train.max_grad_norm,
    weight_decay=config.train.weight_decay,
    eval_strategy="steps",
    eval_steps=config.train.eval_steps,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_steps=config.train.save_epochs * math.ceil(len(train_rows) / config.train.batch_size),  # the Trainer counts steps; one epoch is this many batches
    save_total_limit=config.train.save_limit,
    logging_steps=config.train.log_steps,
    report_to="wandb",  # the Trainer logs into the run opened below
    run_name=config.paths.run_dir.name,
    remove_unused_columns=False,
    dataloader_num_workers=config.train.dataloader_workers,
    dataloader_pin_memory=device == "cuda",
    seed=config.train.seed,
  )

  callbacks = [Best(config.paths.run_dir)]
  if config.train.simulate_epochs: callbacks.append(simulate.Callback(config, retriever, config.train.simulate_epochs, config.train.simulate_episodes))
  trainer = RagActTrainer(
    model=model,
    args=arguments,
    data_collator=collate,
    train_dataset=train_rows,
    eval_dataset=eval_rows,
    callbacks=callbacks,
    retriever=retriever,
    pad_weight=config.train.pad_weight,
    exclude_own_episode=config.train.exclude_own_episode,
  )

  # the run folder; simulate rebuilds the model from its config.yaml, whatever the yaml says later
  run_dir = config.paths.run_dir
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))

  # resuming picks up the latest checkpoint and the same W&B run; None starts fresh
  latest = get_last_checkpoint(run_dir) if config.train.resume else None  # resume_from_checkpoint=True would raise
  wandb_id = run_dir / "wandb_id"
  run_id = wandb_id.read_text().strip() if config.train.resume and wandb_id.exists() else None  # None lets W&B or a sweep agent choose

  rows = {"train": len(train_rows), "eval": len(eval_rows)}
  with wandb.init(
    id=run_id,
    entity=config.wandb.entity,
    project=config.wandb.project,
    mode=config.wandb.mode,
    dir=run_dir.parent,  # every run under one runs/wandb, so `wandb leet runs/wandb` lists them together
    name=run_dir.name,
    config={**config.to_dict(), "rows": rows},
    resume="allow",
  ) as run:
    wandb_id.write_text(run.id)  # simulate reattaches to this run
    run.define_metric("eval/loss", summary="min")  # the summary keeps each run's best, so runs sort and filter by it
    run.define_metric("simulate/training_success", summary="max")
    run.define_metric("simulate/training_success_rate", summary="max")

    trainer.train(resume_from_checkpoint=latest)  # report_to="wandb" logs into the open run
    trainer.save_model(str(run_dir / "final"))
    trainer.state.save_to_json(str(run_dir / "final" / "trainer_state.json"))  # step and epoch, as every checkpoint-* has them

  if config.train.simulate_after.episodes:
    for checkpoint in config.train.simulate_after.checkpoints:
      simulate.main(SimpleNamespace(
        run_dir=run_dir,
        checkpoint=checkpoint,
        config=name,
        episodes=config.train.simulate_after.episodes,
        max_steps=simulate.MAX_STEPS,
        execution="ensemble",
      ))


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])
