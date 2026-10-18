"""Opens an encoded dataset for training, retrieval or eval, whichever encoder wrote it.

Callers use data.load and never see the old layout; legacy.convert rebuilds it into this one.
"""
import numpy as np
from datasets import concatenate_datasets, load_from_disk

from ragmani.data.legacy import convert

TASK_STRIDE = 1_000_000  # a combined row's episode is task index * TASK_STRIDE + its demo's episode id, so tasks never share one


def load(path):
  """An encoded dataset in ragmani's layout, converted when an old encoder wrote it."""
  dataset = load_from_disk(str(path))
  return dataset if {"episode", "step", "is_pad"} <= set(dataset.column_names) else convert(dataset)


def combine(tasks):
  """Every task's dataset as one, each row's episode offset by its task's index times TASK_STRIDE."""
  parts = [load(task.dataset) for task in tasks]
  return concatenate_datasets([part.remove_columns("episode").add_column("episode", np.asarray(part["episode"], dtype=np.int64) + index * TASK_STRIDE) for index, part in enumerate(parts)])


def split(episode):
  """The task index and its demo's episode id of a combined row's episode."""
  return divmod(int(episode), TASK_STRIDE)


def statistics(dataset, std_floor=1e-2):
  """Mean and std of state and of action over every step; std floored as ACT floors it."""
  rows = dataset.with_format("torch", columns=["state", "action"])[:]
  columns = {
    "state": rows["state"],
    "action": rows["action"][:, 0],  # each row's own step; the rest of its chunk repeats later rows' actions
  }
  return {name: {"mean": values.mean(dim=0), "std": values.std(dim=0).clamp(min=std_floor)} for name, values in columns.items()}
