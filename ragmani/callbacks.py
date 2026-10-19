"""Trainer callbacks for training's bookkeeping: the best checkpoint by eval loss, a clean stop from the W&B UI, and W&B alerts."""
import math

from safetensors.torch import save_model
from transformers import TrainerCallback

import wandb


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
