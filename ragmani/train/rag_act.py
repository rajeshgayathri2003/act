"""Trains RagAct on encoded demos with the Hugging Face Trainer, as rag_model.py did.

The loss and the truth schedule live in compute_loss, which reads training progress from the Trainer's own state.
Run from act/ as `python -m ragmani.train.rag_act <config> [section.key=value ...]`.
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from safetensors.torch import save_model
from transformers import Trainer, TrainerCallback, TrainingArguments

from ragmani import data, simulate
from ragmani.config import load
from ragmani.data.retriever import Retriever
from ragmani.data.splits import SPLITS
from ragmani.models import RagAct

COLUMNS = ["image_emb", "state", "next_state", "action", "is_pad", "episode"]
COINS = {
  "row": lambda batch: batch,  # each row its own coin
  "batch": lambda batch: 1,    # one coin for the whole batch, as rag_model.py's add_gt
}


def collate(rows):
  """Stack dataset rows into one batch of tensors."""
  return {key: torch.stack([row[key] for row in rows]) for key in COLUMNS}


class Best(TrainerCallback):
  """Keeps <run_dir>/best as the model at its lowest eval loss so far, with its step and epoch."""

  def __init__(self, run_dir):
    self.directory, self.lowest = run_dir / "best", math.inf

  def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
    if metrics["eval_loss"] >= self.lowest: return
    self.lowest = metrics["eval_loss"]
    self.directory.mkdir(parents=True, exist_ok=True)
    save_model(model, str(self.directory / "model.safetensors"))
    state.save_to_json(str(self.directory / "trainer_state.json"))


class RagActTrainer(Trainer):
  """Runs RagAct with the retriever; training adds the truth with probability 1 - progress, per row or per batch."""

  def __init__(self, *args, retriever, pad_weight, exclude_own_episode, truth_coin, **kwargs):
    super().__init__(*args, **kwargs)
    self.retriever, self.pad_weight, self.exclude_own_episode, self.truth_coin = retriever, pad_weight, exclude_own_episode, COINS[truth_coin]

  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    truth, add_truth = None, None
    if model.training:
      batch, progress = len(inputs["state"]), self.state.global_step / max(self.state.max_steps, 1)
      truth, add_truth = (inputs["action"], inputs["next_state"]), (torch.rand(self.truth_coin(batch), device=inputs["state"].device) < 1 - progress).expand(batch)

    exclude = inputs["episode"] if self.exclude_own_episode else None
    action, is_pad = model(inputs["image_emb"], inputs["state"], self.retriever, exclude=exclude, truth=truth, add_truth=add_truth)
    real = (~inputs["is_pad"]).unsqueeze(-1).float()
    l1 = (F.l1_loss(action, inputs["action"], reduction="none") * real).sum() / (real.sum() * action.shape[-1]).clamp(min=1)  # real steps only
    loss = l1 + self.pad_weight * F.binary_cross_entropy_with_logits(is_pad, inputs["is_pad"].float())
    return (loss, {"action": action, "is_pad": is_pad}) if return_outputs else loss

  def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    """Eval loss through compute_loss; the default calls model(**inputs), which needs a labels column we do not have."""
    with torch.no_grad(): loss = self.compute_loss(model, self._prepare_inputs(inputs))
    return loss.detach(), None, None


def main(name, overrides):
  config = load(name, overrides)
  dataset = data.load(config.paths.dataset).with_format("torch", columns=COLUMNS)
  is_held_out = SPLITS[config.train.split](dataset["episode"][:].numpy(), config.train.eval_fraction, config.train.split_seed)
  train_rows, eval_rows = dataset.select(np.flatnonzero(~is_held_out)), dataset.select(np.flatnonzero(is_held_out))
  if not len(eval_rows): raise ValueError(f"train.eval_fraction={config.train.eval_fraction} holds out no rows of this dataset under the {config.train.split} split")

  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  retriever = Retriever(train_rows, config.retrieval, device)  # memory is training rows only
  model = RagAct(config.model, image_dim=train_rows[0]["image_emb"].shape[-1])
  print(f"{config.train.split} split: {len(train_rows)} training rows, {len(eval_rows)} eval rows; own episode excluded from retrieval: {config.train.exclude_own_episode}; {sum(parameter.numel() for parameter in model.parameters())} parameters")

  arguments = TrainingArguments(
    output_dir=str(config.paths.run_dir),
    num_train_epochs=config.train.epochs,
    per_device_train_batch_size=config.train.batch_size,
    per_device_eval_batch_size=config.train.batch_size,
    learning_rate=config.train.lr,
    lr_scheduler_type=config.train.lr_scheduler,
    lr_scheduler_kwargs=dict(config.train.plateau) if config.train.lr_scheduler == "reduce_lr_on_plateau" else {},
    eval_strategy="steps",
    eval_steps=config.train.eval_steps,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_steps=config.train.save_epochs * math.ceil(len(train_rows) / config.train.batch_size),  # the Trainer counts steps; one epoch is this many batches    logging_steps=config.train.log_steps,
    report_to="wandb",  # the Trainer logs into the run opened below
    run_name=config.paths.run_dir.name,
    remove_unused_columns=False,
    dataloader_num_workers=config.train.dataloader_workers,
    dataloader_pin_memory=device == "cuda",
    seed=config.train.seed,
  )
  trainer = RagActTrainer(
    model=model, args=arguments, data_collator=collate, train_dataset=train_rows, eval_dataset=eval_rows,
    retriever=retriever, pad_weight=config.train.pad_weight, exclude_own_episode=config.train.exclude_own_episode, truth_coin=config.train.truth_coin,
    callbacks=[Best(config.paths.run_dir), *([simulate.Callback(config, retriever, config.train.simulate_epochs)] if config.train.simulate_epochs else [])],
  )

  run_dir = config.paths.run_dir
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))  # simulate rebuilds the model from this, whatever the yaml says later
  rows = {"train": len(train_rows), "eval": len(eval_rows)}
  with wandb.init(entity=config.wandb.entity, project=config.wandb.project, mode=config.wandb.mode, dir=run_dir, name=run_dir.name, config={**config.to_dict(), "rows": rows}) as run:
    (run_dir / "wandb_id").write_text(run.id)  # simulate reattaches to this run
    run.define_metric("eval/loss", summary="min")  # the summary keeps each run's best, so runs sort and filter by it
    run.define_metric("simulate/training_success", summary="max")
    trainer.train()  # report_to="wandb" logs into the open run
    trainer.save_model(str(run_dir / "final"))
    trainer.state.save_to_json(str(run_dir / "final" / "trainer_state.json"))  # step and epoch, as every checkpoint-* has them


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])
