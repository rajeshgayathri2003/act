"""Trains RagAct on encoded demos with the Hugging Face Trainer.

The loss lives in compute_loss: the action L1 on the demo's real steps plus the end-of-demo BCE.
Run from act/ as `uv run python -m ragmani.train <preset> [--section.key value ...]`; `<preset> --help` lists every setting.
"""
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_model
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import wandb
from ragmani import data, simulate, wandb_metrics
from ragmani.config import Config, cli, to_dict
from ragmani.data.retriever import Retriever
from ragmani.models import RagAct

COLUMNS = ["image_emb", "state", "next_state", "action", "is_pad", "episode", "task"]


def retrieval_metrics(idx, inputs, retriever, action, exclude, tasks):
  """How useful and how varied the retrieved rows are: the two scores are 1 for the rows the current frame's own image embedding finds and 0 for no help, and norm_entropy is how many different rows fill the slots."""
  truth_image, mem_image = inputs["image_emb"].float(), retriever.rows["image_emb"].float()
  image_idx = retriever.search(truth_image[:, None], idx.shape[1], exclude).flatten(1)  # the rows a search with the current frame's own image embedding returns: the yardstick the queries' rows are scored against
  truth_action = action.standardize(inputs["action"][:, 0])
  action_distance = lambda rows: (action.standardize(retriever.rows["action"][rows][:, :, 0]) - truth_action[:, None]).abs().mean()
  cos_sim = lambda rows: (mem_image[rows] * truth_image[:, None]).sum(dim=-1).mean()
  mean_action_distance, avg_cos_sim = truth_action.abs().mean(), (truth_image @ mem_image.mean(dim=0)).mean()  # the mean action is 0 once standardized
  row_share = idx.flatten().bincount().float()
  row_share = row_share[row_share > 0] / idx.numel()  # each retrieved row's share of all slots
  task_match = (retriever.rows["task"][idx] == inputs["task"][:, None]).float().mean(-1)  # per frame, the share of its rows from its own task
  task_match = [task_match[inputs["task"] == index].mean() for index in range(len(tasks))]  # nan for a task missing from the batch
  return {
    **({"retrieval/task_match": torch.stack(task_match).nanmean().item(), **{wandb_metrics.key("retrieval", tasks, task, "task_match"): value.item() for task, value in zip(tasks, task_match)}} if len(tasks) > 1 else {}),  # the combined counts each task equally
    "retrieval/action_score": ((mean_action_distance - action_distance(idx)) / (mean_action_distance - action_distance(image_idx))).item(),  # 0 is no better than the mean action
    "retrieval/cos_sim_score": ((cos_sim(idx) - avg_cos_sim) / (cos_sim(image_idx) - avg_cos_sim)).item(),  # 0 is no closer than an average row
    "retrieval/norm_entropy": (-(row_share * row_share.log()).sum() / math.log(idx.numel())).item(),  # Shannon entropy of which row fills each slot over its maximum: 1 all different, 0 one row
  }


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
  """Runs RagAct with the retriever: the loss is the action L1 on the demo's real steps plus the end-of-demo BCE."""

  def __init__(self, *args, retriever, c: Config, **kwargs):
    super().__init__(*args, **kwargs)
    self.retriever = retriever
    self.c = c

  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if model.training:
      progress = self.state.global_step / max(self.state.max_steps, 1)  # 0 at the start of the run, 1 at the end
      model.combined_mask_rate.fill_(min(max((self.c.train.combined_warmup_by - progress) / self.c.train.combined_warmup, 0), 1) if self.c.train.combined_warmup else float(progress < self.c.train.combined_start_at))
    pred_action, pred_pad, idx = model(inputs["image_emb"], inputs["state"], self.retriever, exclude=inputs["episode"]) # the search skips rows from its demo

    target_action = model.action.standardize(inputs["action"])
    real_mask = (~inputs["is_pad"]).unsqueeze(-1).float()  # 1 on chunk steps before the demo's end
    error = F.l1_loss(pred_action, target_action, reduction="none") * real_mask
    losses = {
      "action_l1": error.sum() / (real_mask.sum() * pred_action.shape[-1]).clamp(min=1),  # mean over real steps only
      "pad_bce": F.binary_cross_entropy_with_logits(pred_pad, inputs["is_pad"].float()),
    }

    if not model.training:
      self.eval_rows += len(inputs["state"])
      for name, value in losses.items(): self.eval_losses[name] += value.item() * len(inputs["state"])
    elif self.state.global_step % self.args.logging_steps == 0:
      wandb.log({
        "train/global_step": self.state.global_step,
        **{f"train/{name}": value.item() for name, value in losses.items()},
        **retrieval_metrics(idx, inputs, self.retriever, model.action, inputs["episode"], self.c.tasks),
        **{name: value.item() for name, value in model.diagnostics.items()},
      })

    loss = losses["action_l1"] + self.c.train.pad_weight * losses["pad_bce"]
    return (loss, {"action": pred_action, "is_pad": pred_pad}) if return_outputs else loss

  def evaluate(self, *args, **kwargs):
    """The Trainer's eval, plus the two parts of the eval loss logged on their own."""
    self.eval_losses, self.eval_rows = defaultdict(float), 0
    metrics = super().evaluate(*args, **kwargs)
    wandb.log({"train/global_step": self.state.global_step, **{f"eval/{name}": total / self.eval_rows for name, total in self.eval_losses.items()}})
    return metrics

  def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    """Eval loss through compute_loss; the default calls model(**inputs), which needs a labels column we do not have."""
    with torch.no_grad(): 
      loss = self.compute_loss(model, self._prepare_inputs(inputs))
    return loss.detach(), None, None


def main(c):

  # train and eval rows
  dataset = data.combine(c.tasks).with_format("torch", columns=COLUMNS)
  is_held_out = data.SPLITS[c.train.split](dataset["episode"][:].numpy(), c.train.eval_fraction, c.train.seed)
  train_rows = dataset.select(np.flatnonzero(~is_held_out))
  eval_rows = dataset.select(np.flatnonzero(is_held_out))
  if not len(eval_rows): raise ValueError(f"train.eval_fraction={c.train.eval_fraction} holds out no rows of this dataset under the {c.train.split} split")

  # retrieval memory and model
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  retriever = Retriever(train_rows, c.retrieval, device)  # memory is training rows only
  rows = {key: value.to(device) for key, value in dataset[:].items()}  # every row on the GPU once, so a batch is an index into them
  c.model.d_image = train_rows[0]["image_emb"].shape[-1]
  model = RagAct(c.model)

  if c.model.normalize:
    stats = data.statistics(train_rows)
    model.state.mean, model.state.std = stats["state"]["mean"], stats["state"]["std"]
    model.action.mean, model.action.std = stats["action"]["mean"], stats["action"]["std"]

  print(
    f"{c.train.split} split: {len(train_rows)} training rows, {len(eval_rows)} eval rows; "
    f"{sum(parameter.numel() for parameter in model.parameters())} parameters"
  )

  steps_per_epoch = math.ceil(len(train_rows) / c.train.batch_size)  # the Trainer counts steps
  arguments = TrainingArguments(
    output_dir=str(c.run_dir),
    num_train_epochs=c.train.epochs,
    per_device_train_batch_size=c.train.batch_size,
    per_device_eval_batch_size=c.train.batch_size,
    learning_rate=c.train.lr,
    lr_scheduler_type=c.train.lr_scheduler,
    lr_scheduler_kwargs=c.train.lr_scheduler_kwargs,
    warmup_steps=c.train.warmup_ratio,  # transformers 5 reads a float below 1 as a share of the steps
    max_grad_norm=c.train.max_grad_norm,
    weight_decay=c.train.weight_decay,
    eval_strategy="steps",
    eval_steps=c.train.eval_epochs * steps_per_epoch,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_steps=c.train.save_epochs * steps_per_epoch,
    save_total_limit=c.train.save_limit,
    logging_steps=c.train.log_steps,
    report_to="wandb",  # the Trainer logs into the run opened below
    run_name=c.run_dir.name,
    remove_unused_columns=False,
    dataloader_pin_memory=False,  # batches are already on the device
    seed=c.train.seed,
  )

  stop = wandb_metrics.Stop()
  callbacks = [Best(c.run_dir), stop]
  alerts = {"Gradient norm exploded": wandb_metrics.Alerts.warn_gradient_exploded(c.wandb.alert_grad_norm)} if c.wandb.alert_grad_norm else {}
  if alerts: callbacks.append(wandb_metrics.Alerts(alerts))
  if c.train.simulate_epochs: callbacks.append(simulate.Callback(c, retriever, eval_rows, c.train.simulate_epochs, c.train.simulate_episodes))
  trainer = RagActTrainer(
    model=model,
    args=arguments,
    data_collator=lambda indices: {key: value[torch.as_tensor(indices, device=device)] for key, value in rows.items()},
    train_dataset=np.flatnonzero(~is_held_out).tolist(),  # row numbers; the collator gathers each batch on the device
    eval_dataset=np.flatnonzero(is_held_out).tolist(),
    callbacks=callbacks,
    retriever=retriever,
    c=c,
  )

  # the run folder; simulate rebuilds the model from the c W&B stores, whatever the presets say later
  run_dir = c.run_dir
  run_dir.mkdir(parents=True, exist_ok=True)

  # resuming picks up the latest checkpoint and the same W&B run
  latest = get_last_checkpoint(run_dir) if c.train.resume else None  # resume_from_checkpoint=True would raise
  with wandb_metrics.start(c, run_dir, c.train.resume, config={**to_dict(c), "rows": {"train": len(train_rows), "eval": len(eval_rows)}}, stop=stop):
    trainer.train(resume_from_checkpoint=latest)  # report_to="wandb" logs into the open run
    trainer.save_model(str(run_dir / "final"))
    trainer.state.save_to_json(str(run_dir / "final" / "trainer_state.json"))  # step and epoch, as every checkpoint-* has them

  if c.train.simulate_after.episodes and not stop.requested:
    # simulate in a fresh process: training's GPU physics and CUDA memory are released only when its process ends, about 4 GB
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-m", "ragmani.simulate", str(run_dir), "--config", c.name, "--episodes", str(c.train.simulate_after.episodes), "--checkpoint", *c.train.simulate_after.checkpoints])


if __name__ == "__main__":
  main(cli(description=__doc__.splitlines()[0]))
