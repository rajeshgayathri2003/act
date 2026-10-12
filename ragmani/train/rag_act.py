"""Trains RagAct on encoded demos with the Hugging Face Trainer, as rag_model.py does.

The loss and the truth schedule live in compute_loss, which reads training progress from the Trainer's own state.
Run from act/ as `python -m ragmani.train.rag_act <config> [section.key=value ...]`.
"""
import math
import signal
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
from ragmani.models.policies import standardize

COLUMNS = ["image_emb", "state", "next_state", "action", "is_pad", "episode"]
TRUTH_SAMPLING = {  # how many random numbers decide which rows get the truth row
  "per_row": lambda batch: batch,
  "per_batch": lambda batch: 1,  # the whole batch gets it or none does, as rag_model.py
}
TRUTH_SCHEDULES = {  # progress from 0 to 1; the truth row is added with probability 1 - progress
  "whole_run": lambda state, schedule: state.global_step / max(state.max_steps, 1),  # as rag_model.py
  "fixed_epochs": lambda state, schedule: state.epoch / schedule.epochs,                 # the same at a given epoch whatever the run's length
  "stepped": lambda state, schedule: state.epoch // schedule.step_epochs * schedule.step_epochs / schedule.epochs,  # as fixed_epochs, held flat between drops every step_epochs
}


def retrieval(indices, inputs, retriever):
  """How varied and how close the retrieved rows are: share of distinct rows, and mean similarity to the row's own frame."""
  return {
    "retrieval/distinct": len(indices.unique()) / indices.numel(),
    "retrieval/similarity": (retriever.rows["image_emb"][indices] * inputs["image_emb"][:, None]).sum(dim=-1).mean().item(),
  }


def collate(rows):
  """Stack dataset rows into one batch of tensors."""
  return {key: torch.stack([row[key] for row in rows]) for key in COLUMNS}


class Best(TrainerCallback):
  """Keeps <run_dir>/best as the model at its lowest eval loss so far, with its step and epoch."""

  def __init__(self, run_dir):
    self.directory, self.lowest = run_dir / "best", math.inf

  def on_train_begin(self, args, state, control, **kwargs):
    if state.best_metric is not None: self.lowest = state.best_metric  # a resumed run's state carries its best eval loss, so best/ is not overwritten by a worse model

  def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
    if metrics["eval_loss"] >= self.lowest: return
    self.lowest = metrics["eval_loss"]
    self.directory.mkdir(parents=True, exist_ok=True)
    save_model(model, str(self.directory / "model.safetensors"))
    state.save_to_json(str(self.directory / "trainer_state.json"))


class RagActTrainer(Trainer):
  """Runs RagAct with the retriever; training adds the truth with probability 1 - progress, per row or per batch."""

  def __init__(self, *args, retriever, pad_weight, exclude_own_episode, truth_sampling, truth_schedule, **kwargs):
    super().__init__(*args, **kwargs)
    self.retriever, self.pad_weight, self.exclude_own_episode, self.truth_sampling, self.truth_schedule = retriever, pad_weight, exclude_own_episode, TRUTH_SAMPLING[truth_sampling], truth_schedule

  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    truth, add_truth = None, None
    if model.training:
      batch, progress = len(inputs["state"]), TRUTH_SCHEDULES[self.truth_schedule.mode](self.state, self.truth_schedule)
      truth, add_truth = (inputs["action"], inputs["next_state"]), (torch.rand(self.truth_sampling(batch), device=inputs["state"].device) < 1 - progress).expand(batch)

    exclude = inputs["episode"] if self.exclude_own_episode else None
    action, is_pad, indices = model(inputs["image_emb"], inputs["state"], self.retriever, exclude=exclude, truth=truth, add_truth=add_truth)
    real = (~inputs["is_pad"]).unsqueeze(-1).float()
    parts = {
      "action_l1": (F.l1_loss(action, standardize(model.action_scale, inputs["action"]), reduction="none") * real).sum() / (real.sum() * action.shape[-1]).clamp(min=1),  # real steps only
      "pad_bce": F.binary_cross_entropy_with_logits(is_pad, inputs["is_pad"].float()),
    }
    if not model.training:
      for name, value in parts.items(): self.eval_parts[name].append(value.detach())
    elif self.state.global_step % self.args.logging_steps == 0:
      wandb.log({"train/global_step": self.state.global_step, **{f"train/{name}": value.item() for name, value in parts.items()}, **retrieval(indices, inputs, self.retriever)})

    loss = parts["action_l1"] + self.pad_weight * parts["pad_bce"]
    return (loss, {"action": action, "is_pad": is_pad}) if return_outputs else loss

  def evaluate(self, *args, **kwargs):
    """The Trainer's eval, plus the two parts of the eval loss logged on their own."""
    self.eval_parts = defaultdict(list)
    metrics = super().evaluate(*args, **kwargs)
    wandb.log({"train/global_step": self.state.global_step, **{f"eval/{name}": torch.stack(values).mean().item() for name, values in self.eval_parts.items()}})
    return metrics

  def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    """Eval loss through compute_loss; the default calls model(**inputs), which needs a labels column we do not have."""
    with torch.no_grad(): loss = self.compute_loss(model, self._prepare_inputs(inputs))
    return loss.detach(), None, None


def preempted(signum, frame):
  """On SIGTERM (a Job's pod being stopped), mark the W&B run preempted and exit non-zero, so a sweep re-queues it."""
  if wandb.run: wandb.run.mark_preempting()
  sys.exit(1)


def main(name, overrides):
  signal.signal(signal.SIGTERM, preempted)
  test = next((override.partition("=")[2] for override in overrides if override.startswith("test=")), None)  # a sweep's one change, e.g. test=train.lr=1e-5
  config = load(name, [override for override in overrides if not override.startswith("test=")] + ([test] if test not in (None, "unchanged") else []))
  if test: config.paths.run_dir = config.paths.run_dir / test  # each test of a sweep in its own folder
  dataset = data.load(config.paths.dataset).with_format("torch", columns=COLUMNS)
  is_held_out = SPLITS[config.train.split](dataset["episode"][:].numpy(), config.train.eval_fraction, config.train.split_seed)
  train_rows, eval_rows = dataset.select(np.flatnonzero(~is_held_out)), dataset.select(np.flatnonzero(is_held_out))
  if config.model.query == "observation" and not config.train.exclude_own_episode: raise ValueError("model.query=observation needs train.exclude_own_episode=true, or a row's search finds itself and its own demo")
  if not len(eval_rows): raise ValueError(f"train.eval_fraction={config.train.eval_fraction} holds out no rows of this dataset under the {config.train.split} split")

  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  retriever = Retriever(train_rows, config.retrieval, device)  # memory is training rows only
  model = RagAct(config.model, image_dim=train_rows[0]["image_emb"].shape[-1])
  if config.model.normalize: 
    scales = data.statistics(train_rows)
    model.state_scale, model.action_scale = scales["state"], scales["action"]
  print(f"{config.train.split} split: {len(train_rows)} training rows, {len(eval_rows)} eval rows; own episode excluded from retrieval: {config.train.exclude_own_episode}; {sum(parameter.numel() for parameter in model.parameters())} parameters")

  arguments = TrainingArguments(
    output_dir=str(config.paths.run_dir),
    num_train_epochs=config.train.epochs,
    per_device_train_batch_size=config.train.batch_size,
    per_device_eval_batch_size=config.train.batch_size,
    learning_rate=config.train.lr,
    lr_scheduler_type=config.train.lr_scheduler,
    lr_scheduler_kwargs=dict(config.train.plateau) if config.train.lr_scheduler == "reduce_lr_on_plateau" else {},
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
  trainer = RagActTrainer(
    model=model, args=arguments, data_collator=collate, train_dataset=train_rows, eval_dataset=eval_rows,
    retriever=retriever, pad_weight=config.train.pad_weight, exclude_own_episode=config.train.exclude_own_episode, truth_sampling=config.train.truth_sampling, truth_schedule=config.train.truth_schedule,
    callbacks=[Best(config.paths.run_dir), *([simulate.Callback(config, retriever, config.train.simulate_epochs, config.train.simulate_episodes)] if config.train.simulate_epochs else [])],
  )

  run_dir = config.paths.run_dir
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))  # simulate rebuilds the model from this, whatever the yaml says later
  rows = {"train": len(train_rows), "eval": len(eval_rows)}
  latest = get_last_checkpoint(run_dir) if config.train.resume else None  # None starts fresh; resume_from_checkpoint=True would raise
  wandb_id = run_dir / "wandb_id"
  with wandb.init(**({"id": wandb_id.read_text().strip()} if config.train.resume and wandb_id.exists() else {}), entity=config.wandb.entity, project=config.wandb.project, mode=config.wandb.mode, dir=run_dir.parent, name=run_dir.name, config={**config.to_dict(), "rows": rows}, resume="allow") as run:  # every run under one runs/wandb, so `wandb leet runs/wandb` lists them together
    (run_dir / "wandb_id").write_text(run.id)  # simulate reattaches to this run
    run.define_metric("eval/loss", summary="min")  # the summary keeps each run's best, so runs sort and filter by it
    run.define_metric("simulate/training_success", summary="max")
    run.define_metric("simulate/training_success_rate", summary="max")
    trainer.train(resume_from_checkpoint=latest)  # report_to="wandb" logs into the open run
    trainer.save_model(str(run_dir / "final"))
    trainer.state.save_to_json(str(run_dir / "final" / "trainer_state.json"))  # step and epoch, as every checkpoint-* has them

  for checkpoint in config.train.simulate_after.checkpoints if config.train.simulate_after.episodes else []:
    simulate.main(SimpleNamespace(run_dir=run_dir, checkpoint=checkpoint, config=name, episodes=config.train.simulate_after.episodes, max_steps=simulate.MAX_STEPS, execution="ensemble"))


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])
