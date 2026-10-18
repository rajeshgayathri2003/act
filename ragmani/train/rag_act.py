"""Trains RagAct on encoded demos with the Hugging Face Trainer, as rag_model.py does.

The loss and the truth schedule live in compute_loss, which reads training progress from the Trainer's own state.
Run from act/ as `uv run python -m ragmani.train.rag_act <config> [section.key=value ...]`.
"""
import math
import os
import sys
from collections import defaultdict

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
ACTION_TEMPERATURE = 0.1  # the retrieval loss's target: candidates whose action is this much further from the truth's get e times less weight
SCORE_TEMPERATURE = 0.01  # divides the cosine of q_fetch and each candidate; the embeddings are close together, so a small one spreads the scores


def retrieval_metrics(idx, inputs, retriever, action, exclude):
  """How useful and how varied the retrieved rows are: the two scores are 1 for the rows the current frame's own image embedding finds and 0 for no help, and norm_entropy is how many different rows fill the slots."""
  truth_image, mem_image = inputs["image_emb"].float(), retriever.rows["image_emb"].float()
  image_idx = retriever.search(truth_image[:, None], idx.shape[1], exclude).flatten(1)  # the rows a search with the current frame's own image embedding returns: the yardstick the queries' rows are scored against
  truth_action = action.standardize(inputs["action"][:, 0])
  action_distance = lambda rows: (action.standardize(retriever.rows["action"][rows][:, :, 0]) - truth_action[:, None]).abs().mean()
  cos_sim = lambda rows: (mem_image[rows] * truth_image[:, None]).sum(dim=-1).mean()
  mean_action_distance, avg_cos_sim = truth_action.abs().mean(), (truth_image @ mem_image.mean(dim=0)).mean()  # the mean action is 0 once standardized
  row_share = idx.flatten().bincount().float()
  row_share = row_share[row_share > 0] / idx.numel()  # each retrieved row's share of all slots
  return {
    "retrieval/action_score": ((mean_action_distance - action_distance(idx)) / (mean_action_distance - action_distance(image_idx))).item(),  # 0 is no better than the mean action
    "retrieval/cos_sim_score": ((cos_sim(idx) - avg_cos_sim) / (cos_sim(image_idx) - avg_cos_sim)).item(),  # 0 is no closer than an average row
    "retrieval/norm_entropy": (-(row_share * row_share.log()).sum() / math.log(idx.numel())).item(),  # Shannon entropy of which row fills each slot over its maximum: 1 all different, 0 one row
  }


def retrieval_loss(q_fetch, idx, inputs, retriever, action, exclude):
  """Cross-entropy teaching q_fetch to rank candidate rows by how close their first action is to the frame's true one: each query's own kept rows, the rows the frame's own image embedding finds, and random rows."""
  batch, queries, _ = q_fetch.shape
  count = idx.shape[1]
  query_idx = idx.view(batch, queries, -1)  # the rows each query's own search kept
  image_idx = retriever.search(inputs["image_emb"][:, None].float(), count, exclude).view(batch, 1, count).expand(-1, queries, -1)  # the rows a search with the current frame's own image embedding returns, shared by its queries
  random_idx = torch.randint(0, len(retriever.rows["image_emb"]), (batch, queries, count), device=idx.device)  # easy negatives
  candidates = torch.cat([query_idx, image_idx, random_idx], dim=-1)  # (B, queries, candidates): every query ranks all of them against how close each row's action is to the frame's true one
  q_fetch_direction = F.normalize(q_fetch, dim=-1, eps=0.05)
  scores = (q_fetch_direction[:, :, None] * retriever.rows["image_emb"][candidates].float()).sum(dim=-1) / SCORE_TEMPERATURE  # cosine, since the embeddings are unit length
  truth_action = action.standardize(inputs["action"][:, 0])
  distance = (action.standardize(retriever.rows["action"][candidates][..., 0, :]) - truth_action[:, None, None]).abs().mean(dim=-1)
  own = retriever.rows["episode"][candidates] == exclude[:, None, None] if exclude is not None else torch.zeros_like(candidates, dtype=torch.bool)  # random rows from the frame's own demo, which the search never returns
  target = torch.softmax(-distance.masked_fill(own, math.inf) / ACTION_TEMPERATURE, dim=-1)
  return -(target * torch.log_softmax(scores.masked_fill(own, -math.inf), dim=-1).masked_fill(own, 0)).sum(dim=-1).mean()


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


class WandbStop(TrainerCallback):
  """Ends training after the current step when the run is stopped from the W&B UI, so the process exits cleanly and a Job does not restart it."""

  def __init__(self):
    self.requested = False

  def request(self): self.requested = True

  def on_step_end(self, args, state, control, **kwargs):
    if self.requested: control.should_training_stop = True


class WandbAlerts(TrainerCallback):
  """Sends a W&B alert, at most hourly per rule, whenever a rule returns a message for the logged metrics; training continues."""

  def __init__(self, rules):
    self.rules = rules  # {title: rule(logs) -> message or None}

  def on_log(self, args, state, control, logs=None, **kwargs):
    for title, rule in self.rules.items():
      if message := rule(logs or {}):
        wandb.alert(title=title, text=f"{args.run_name}, step {state.global_step}, epoch {state.epoch:.1f}: {message}", level=wandb.AlertLevel.WARN, wait_duration=3600)

  @staticmethod
  def warn_gradient_exploded(limit, streak=3):
    """A rule: fires when the gradient norm is not finite, or above limit for streak logs in a row."""
    high = 0

    def rule(logs):
      nonlocal high
      if "grad_norm" not in logs: return None
      norm = float(logs["grad_norm"])
      high = high + 1 if not math.isfinite(norm) or norm > limit else 0
      return f"grad norm {norm:.3g}" if not math.isfinite(norm) or high >= streak else None
    return rule


class RagActTrainer(Trainer):
  """Runs RagAct with the retriever; training gives each row its truth row with probability 1 - progress through the run."""

  def __init__(self, *args, retriever, pad_weight, exclude_own_episode, truth_row, retrieval_weight, **kwargs):
    super().__init__(*args, **kwargs)
    self.retriever = retriever
    self.pad_weight = pad_weight
    self.exclude_own_episode = exclude_own_episode
    self.truth_row = truth_row
    self.retrieval_weight = retrieval_weight

  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    truth, add_truth, add_image_emb = None, None, None
    progress = self.state.global_step / max(self.state.max_steps, 1)  # 0 at the start of the run, 1 at the end
    if model.training and model.config.add_image_emb_to_search and model.config.image_fade:
      add_image_emb = torch.rand(len(inputs["state"]), device=inputs["state"].device) < 0.5 * (1 + math.cos(math.pi * min(progress / model.config.image_fade, 1)))  # each row on its own draw: flat at first, steepest midway, none once the fade is over
    if model.training and self.truth_row:
      truth = (inputs["action"], inputs["next_state"])
      add_truth = torch.rand(len(inputs["state"]), device=inputs["state"].device) < 1 - progress  # each row on its own draw

    exclude = inputs["episode"] if self.exclude_own_episode else None
    pred_action, pred_pad, idx = model(inputs["image_emb"], inputs["state"], self.retriever, exclude=exclude, truth=truth, add_truth=add_truth, add_image_emb=add_image_emb)

    target_action = model.action.standardize(inputs["action"])
    real_mask = (~inputs["is_pad"]).unsqueeze(-1).float()  # 1 on chunk steps before the demo's end
    error = F.l1_loss(pred_action, target_action, reduction="none") * real_mask
    losses = {
      "action_l1": error.sum() / (real_mask.sum() * pred_action.shape[-1]).clamp(min=1),  # mean over real steps only
      "pad_bce": F.binary_cross_entropy_with_logits(pred_pad, inputs["is_pad"].float()),
    }
    if self.retrieval_weight: losses["retrieval"] = retrieval_loss(model.q_fetch, idx, inputs, self.retriever, model.action, exclude)  # logged as train/retrieval and eval/retrieval

    if not model.training:
      self.eval_rows += len(inputs["state"])
      for name, value in losses.items(): self.eval_losses[name] += value.item() * len(inputs["state"])
    elif self.state.global_step % self.args.logging_steps == 0:
      wandb.log({"train/global_step": self.state.global_step, **{f"train/{name}": value.item() for name, value in losses.items()}, **retrieval_metrics(idx, inputs, self.retriever, model.action, exclude), **{name: value.item() for name, value in model.diagnostics.items()}})

    loss = losses["action_l1"] + self.pad_weight * losses["pad_bce"] + (self.retrieval_weight * losses.get("retrieval", 0) if model.training else 0)  # eval_loss, which picks best, leaves retrieval out
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
  dataset = data.combine(config.tasks).with_format("torch", columns=COLUMNS)
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
    warmup_steps=config.train.warmup_ratio,  # transformers 5 reads a float below 1 as a share of the steps
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

  stop = WandbStop()
  callbacks = [Best(config.paths.run_dir), stop]
  alerts = {"Gradient norm exploded": WandbAlerts.warn_gradient_exploded(config.train.alert_grad_norm)} if config.train.alert_grad_norm else {}
  if alerts: callbacks.append(WandbAlerts(alerts))
  if config.train.simulate_epochs: callbacks.append(simulate.Callback(config, retriever, eval_rows, config.train.simulate_epochs, config.train.simulate_episodes))
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
    truth_row=config.train.truth_row,
    retrieval_weight=config.train.retrieval_weight,
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
    group=config.wandb.group,
    mode=config.wandb.mode,
    dir=run_dir.parent,  # every run under one runs/wandb, so `wandb leet runs/wandb` lists them together
    name=run_dir.name,
    config={**config.to_dict(), "rows": rows},
    resume="allow",
    settings=wandb.Settings(stop_fn=stop.request),  # the UI's stop ends training instead of interrupting it
  ) as run:
    wandb_id.write_text(run.id)  # simulate reattaches to this run
    run.define_metric("eval/loss", summary="min")  # the summary keeps each run's best, so runs sort and filter by it
    run.define_metric("simulate/training_success", summary="max")
    run.define_metric("simulate/training_success_rate", summary="max")

    trainer.train(resume_from_checkpoint=latest)  # report_to="wandb" logs into the open run
    trainer.save_model(str(run_dir / "final"))
    trainer.state.save_to_json(str(run_dir / "final" / "trainer_state.json"))  # step and epoch, as every checkpoint-* has them

  if config.train.simulate_after.episodes and not stop.requested:
    # simulate in a fresh process: training's GPU physics and CUDA memory are released only when its process ends, about 4 GB
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-m", "ragmani.simulate", str(run_dir), "--config", str(name), "--episodes", str(config.train.simulate_after.episodes), "--checkpoint", *config.train.simulate_after.checkpoints])


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])
