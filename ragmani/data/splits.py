"""Train/eval splits of encoded demos, as masks of held-out rows, chosen by name from SPLITS.

Each takes the per-row episode ids, the held-out fraction and a seed, so splits swap by config and compare on equal terms.
"""
import numpy as np


def episode(episodes, fraction, seed):
  """Hold out whole demos, so no held-out frame has a neighbour in training."""
  unique = np.unique(episodes)
  return np.isin(episodes, np.random.default_rng(seed).choice(unique, int(len(unique) * fraction), replace=False))


def frame(episodes, fraction, seed):
  """Hold out random rows, as rag_model.py does; a held-out frame's neighbours from the same demo can be in training."""
  held_out = np.zeros(len(episodes), dtype=bool)
  held_out[np.random.default_rng(seed).permutation(len(episodes))[:int(len(episodes) * fraction)]] = True
  return held_out


SPLITS = {
  "episode": episode,
  "frame": frame,
}
