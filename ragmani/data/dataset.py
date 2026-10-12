"""Opens an encoded dataset for training, retrieval or eval, whichever encoder wrote it.

Callers use data.load and never see the old layout; legacy.convert rebuilds it into this one.
"""
from datasets import load_from_disk

from ragmani.data.legacy import convert


def load(path):
  """An encoded dataset in ragmani's layout, converted when an old encoder wrote it."""
  dataset = load_from_disk(str(path))
  return dataset if {"episode", "step", "is_pad"} <= set(dataset.column_names) else convert(dataset)


def statistics(dataset, std_floor=1e-2):
  """Mean and std of state and of action over every step; std floored as ACT floors it."""
  rows = dataset.with_format("torch", columns=["state", "action"])[:]
  columns = {
    "state": rows["state"],
    "action": rows["action"][:, 0],  # each row's own step; the rest of its chunk repeats later rows' actions
  }
  return {name: {"mean": values.mean(dim=0), "std": values.std(dim=0).clamp(min=std_floor)} for name, values in columns.items()}
