"""How runs talk to W&B: what each metric means, its x-axis and summary, the run itself, and the W&B Trainer callbacks.

Training and simulate both open their run through start, so the two share one W&B run and one set of metric settings.
"""
import math
from contextlib import contextmanager

from transformers import TrainerCallback

import wandb

STEP, EPOCH = "train/global_step", "train/epoch"

METRICS = {  # name: (meaning, x-axis, summary)
  STEP: ("Optimizer steps so far.", None, None),
  EPOCH: ("Passes over the training data so far.", None, None),
  "train/loss": ("What training minimizes. action_l1 + pad_weight * pad_bce, or action_ce in place of action_l1 with the codebook head.", STEP, None),
  "train/action_l1": ("How far predicted actions are from the demo's; 0 is perfect. Mean |predicted - demo| over the chunk's steps before the demo ends, each action dimension divided by its std.", STEP, None),
  "train/action_ce": ("Codebook head only: how well the predicted bins match the demo's actions. Cross-entropy against each action dimension's two-hot code over the chunk's steps before the demo ends; ln(bins) is uniform guessing.", STEP, None),
  "train/pad_bce": ("How well it predicts that the demo has ended. Binary cross-entropy over every chunk step.", STEP, None),
  "train/grad_norm": ("How big the gradient is; spikes mean unstable training. L2 norm of all gradients, before clipping.", STEP, None),
  "train/learning_rate": ("The learning rate at this step.", STEP, None),
  "eval/loss": ("train/loss on held-out demos the model never trains on. The summary keeps the lowest.", STEP, "min"),
  "eval/action_l1": ("train/action_l1 on held-out demos.", STEP, None),
  "eval/pad_bce": ("train/pad_bce on held-out demos.", STEP, None),
  "eval/action_ce": ("train/action_ce on held-out demos.", STEP, None),
  "eval/runtime": ("Seconds one eval took.", STEP, None),
  "eval/samples_per_second": ("Rows per second during eval.", STEP, None),
  "eval/steps_per_second": ("Batches per second during eval.", STEP, None),
  "model/combined_magnitude": ("How big the frame's features are. Mean L2 length of the image and state MLP outputs joined.", STEP, None),
  "model/q_fetch_magnitude": ("How big the search vectors are; near 0 means the searches collapsed. Mean L2 length of W q.", STEP, None),
  "retrieval/action_score": ("How useful the fetched rows' actions are. 1 = as good as an image search's rows, 0 = no better than the average action, below 0 = worse. From each row's distance to the frame's true action.", STEP, None),
  "retrieval/cos_sim_score": ("How much the fetched rows look like the current frame. 1 = as alike as an image search's rows, 0 = as alike as an average row, below 0 = less alike. Rescaled cosine, since any frame already has about 0.994 cosine with the average row.", STEP, None),
  "retrieval/norm_entropy": ("How many different rows get fetched. 1 = every slot in the batch a different row, 0 = the same row everywhere. Entropy of the row counts over its maximum.", STEP, None),
  "retrieval/attn_entropy": ("How evenly attention spreads over the fetched rows. ln(n_queries * n_keep) = evenly, 0 = all on one row.", STEP, None),
  "retrieval/task_match": ("How often fetched rows come from the frame's own task; 1 = always. Averaged per task, then across tasks so each counts equally.", STEP, None),
  "retrieval/neighbours": ("What the searches fetch: one held-out demo at 20%, 50% and 80% through, each frame followed by each search's nearest row.", EPOCH, None),
  "check/success_rate": ("Share of the training check's layouts solved, all tasks together. The same layouts every check; the summary keeps the highest.", EPOCH, "max"),
  "sim/success_rate": ("Share of layouts each saved checkpoint solves after training, all tasks together. Different layouts from the checks; the summary keeps the highest.", EPOCH, "max"),
  "sim/episodes": ("One row per after-training episode: checkpoint, task, seed, success and steps.", EPOCH, None),
  "train_loss": ("Mean train/loss over the whole run, from the Trainer at the end.", None, None),
  "train_runtime": ("Seconds the whole training took.", None, None),
  "train_samples_per_second": ("Training rows per second over the whole run.", None, None),
  "train_steps_per_second": ("Optimizer steps per second over the whole run.", None, None),
  "total_flos": ("Floating-point operations the Trainer counted over the whole run.", None, None),
}
TASK_METRICS = {  # (section, name): (meaning, x-axis, summary), one per task
  ("check", "success_rate"): ("check/success_rate for this task only.", EPOCH, "max"),
  ("check", "video"): ("One check episode, on the same layout every check; not scored.", EPOCH, None),
  ("check", "grid"): ("Every scored check episode side by side: green border solved, red failed.", EPOCH, None),
  ("sim", "success_rate"): ("sim/success_rate for this task only.", EPOCH, "max"),
  ("sim", "grid"): ("Every after-training episode of this task: green border solved, red failed.", EPOCH, None),
  ("retrieval", "task_match"): ("retrieval/task_match for this task's frames only.", STEP, None),
}


def key(section, tasks, task, name):
  """A W&B key: section/name with one task; with several, in the task's own section, so shared sections hold only combined metrics."""
  return f"{section}/{name}" if len(tasks) == 1 else f"{task.name}/{section}_{name}"


@contextmanager
def start(c, run_dir, attach, config=None, stop=None):
  """The run's W&B run, reattached through run_dir/wandb_id when attach; every metric gets its x-axis and summary, and a new run logs what each means."""
  wandb_id = run_dir / "wandb_id"
  run_id = wandb_id.read_text().strip() if attach and wandb_id.exists() else None
  settings = wandb.Settings(stop_fn=stop.request) if stop else None  # the UI's stop ends training instead of interrupting it
  with wandb.init(id=run_id, entity=c.wandb.entity, project=c.wandb.project, group=c.wandb.group, mode=c.wandb.mode, dir=run_dir.parent, name=run_dir.name, config=config, resume="allow", settings=settings) as run:  # dir: every run under one runs/wandb
    wandb_id.write_text(run.id)  # simulate reattaches to this run
    metrics = {**{key(section, c.tasks, task, name): entry for task in c.tasks for (section, name), entry in TASK_METRICS.items()}, **METRICS}
    if len(c.tasks) == 1: del metrics["retrieval/task_match"]  # one task always matches
    if c.model.action_head != "codebook": del metrics["train/action_ce"], metrics["eval/action_ce"]  # only the codebook head has bin logits
    for name, (_, axis, summary) in metrics.items(): run.define_metric(name, step_metric=axis, summary=summary, hidden=name in (STEP, EPOCH) or None)
    if run_id is None: run.log({"about/metrics": wandb.Table(columns=["metric", "meaning"], data=[[name, meaning] for name, (meaning, _, _) in metrics.items()])})
    yield run


class Stop(TrainerCallback):
  """Ends training after the current step when the run is stopped from the W&B UI, so the process exits cleanly and a Job does not restart it."""

  def __init__(self):
    self.requested = False

  def request(self): self.requested = True

  def on_step_end(self, args, state, control, **kwargs):
    if self.requested: control.should_training_stop = True


class Alerts(TrainerCallback):
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
