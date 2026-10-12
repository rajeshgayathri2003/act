"""Reads ManiSkill demonstrations: an h5 of episodes and the json ManiSkill writes beside it.

Needs only h5py, not the simulator, so data can be inspected on machines without rendering.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class Episode:
  id: int
  actions: np.ndarray   # (T, action_dim)
  env_states: dict      # nested arrays of length T+1: the state before each action, then the final state
  success: np.ndarray   # (T,) bool, per step

  def state(self, step):
    """The env state at one step, as env.set_state_dict takes it."""
    return pick(self.env_states, step)


class Trajectories:
  """The episodes of one demo file, in episode-id order (h5 keys sort as traj_0, traj_1, traj_10)."""

  def __init__(self, path):
    self.path = Path(path)
    info = json.loads(self.path.with_suffix(".json").read_text())
    self.env_id = info["env_info"]["env_id"]
    self.control_mode = info["env_info"]["env_kwargs"]["control_mode"]
    self.ids = [episode["episode_id"] for episode in info["episodes"]]

  def __len__(self): return len(self.ids)

  def __iter__(self):
    with h5py.File(self.path, "r") as file:
      for episode_id in self.ids:
        group = file[f"traj_{episode_id}"]
        yield Episode(id=episode_id,actions=group["actions"][:], env_states=read(group["env_states"]), success=group["success"][:])


def read(group):
  """Load an h5 group into nested dicts of arrays."""
  return {key: read(item) if isinstance(item, h5py.Group) else item[:] for key, item in group.items()}


def pick(tree, step):
  """Index every array in a nested dict at one step."""
  return {key: pick(item, step) if isinstance(item, dict) else item[step] for key, item in tree.items()}
