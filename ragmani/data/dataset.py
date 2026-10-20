"""Encoded datasets for training, retrieval and eval: every task's rows as one, the held-out split, and the stats that standardize them."""
import numpy as np
from datasets import concatenate_datasets, load_from_disk


def episode(episodes, fraction, seed):
  """Hold out whole demos, so no held-out frame has a neighbour in training."""
  unique = np.unique(episodes)
  return np.isin(episodes, np.random.default_rng(seed).choice(unique, int(len(unique) * fraction), replace=False))


def frame(episodes, fraction, seed):
  """Hold out random rows, as rag_model.py does; a held-out frame's neighbours from the same demo can be in training."""
  held_out = np.zeros(len(episodes), dtype=bool)
  held_out[np.random.default_rng(seed).permutation(len(episodes))[:int(len(episodes) * fraction)]] = True
  return held_out


SPLITS = {  # each takes the rows' episode ids, the held-out fraction and a seed, and returns the held-out mask
  "episode": episode,
  "frame": frame,
}


def combine(tasks):
  """Every task's dataset as one, episode ids continuing past the previous task's; demo is the id in the task's own h5."""
  parts, offset = [], 0
  for index, task in enumerate(tasks):
    part = load_from_disk(str(task.dataset))
    demo = np.asarray(part["episode"], dtype=np.int64)
    parts.append(part.remove_columns("episode").add_column("episode", demo + offset).add_column("demo", demo).add_column("task", np.full(len(demo), index)))
    offset += int(demo.max()) + 1
  return concatenate_datasets(parts)


def statistics(dataset, std_floor=1e-2):
  """Mean and std of state and of action over every step; std floored as ACT floors it."""
  rows = dataset.with_format("torch", columns=["state", "action"])[:]
  columns = {
    "state": rows["state"],
    "action": rows["action"][:, 0],  # each row's own step; the rest of its chunk repeats later rows' actions
  }
  return {name: {"mean": values.mean(dim=0), "std": values.std(dim=0).clamp(min=std_floor)} for name, values in columns.items()}
