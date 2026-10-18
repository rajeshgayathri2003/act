"""Replays ManiSkill demos and stores one row per step: the frame's embedding, robot state and the action chunk ahead.

The only data module that imports ManiSkill, since it renders. Run as `python -m ragmani.data.encode <config> [section.key=value ...]`.
"""
import itertools
import sys
from collections import defaultdict

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers the ManiSkill envs with gymnasium
import numpy as np
import torch
from datasets import Array2D, Dataset, Features, List, Value
from tqdm import tqdm

from ragmani.config import load
from ragmani.data.trajectories import Trajectories
from ragmani.models import ENCODERS


def rows(episode, env, encoder, chunk, batch_size):
  """One row per action of the episode, with next_state from the recorded state after it."""
  steps = len(episode.actions)
  qpos, frames = [], []
  env.reset()  # once per episode; set_state_dict then overwrites every actor and the robot

  for step in range(steps + 1):  # T+1 states, so the last action's next_state is the recorded final state
    env.set_state_dict(episode.state(step))
    qpos.append(env.get_obs()["agent"]["qpos"][0].cpu().numpy())
    if step < steps: frames.append(env.render()[0].cpu().numpy())

  embeddings = torch.cat([encoder.embed(frames[start:start + batch_size]) for start in range(0, steps, batch_size)]).numpy()

  for step in range(steps):
    window = episode.actions[step:step + chunk]
    action = np.zeros((chunk, window.shape[1]), np.float32)
    action[:len(window)] = window
    yield {
      "episode": episode.id,
      "step": step,
      "image_emb": embeddings[step],
      "state": qpos[step],
      "next_state": qpos[step + 1],
      "action": action,
      "is_pad": np.arange(chunk) >= len(window),  # past the episode end; the zero rows there are not actions
    }


def main(name, overrides):
  config = load(name, overrides)
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  encoder = ENCODERS[config.encoder.name](config.encoder, device, config.paths.cache)
  for task in config.tasks:
    if task.dataset.exists(): print(f"{task.env_id}: {task.dataset} exists, skipped")
    else: encode(task, config, encoder)


def encode(task, config, encoder):
  """Replays one task's demos and saves its rows to the task's dataset."""
  trajectories = Trajectories(task.traj)
  # replay sets states and renders without stepping, so the controller is unused; joint control needs no IK model
  env = gym.make(trajectories.env_id, num_envs=1, control_mode="pd_joint_pos", obs_mode="state_dict", render_mode="rgb_array").unwrapped  # gymnasium wrappers do not forward set_state_dict
  episodes = itertools.islice(trajectories, config.encode.max_episodes)

  columns = defaultdict(list)
  for episode in tqdm(episodes, total=config.encode.max_episodes or len(trajectories)):
    for row in rows(episode, env, encoder, config.model.chunk, config.encode.batch_size):
      for key, value in row.items(): columns[key].append(value)

  state_dim, action_dim = len(columns["state"][0]), columns["action"][0].shape[1]
  features = Features({
    "episode": Value("int32"),
    "step": Value("int32"),
    "image_emb": List(Value("float32"), length=encoder.dim),
    "state": List(Value("float32"), length=state_dim),
    "next_state": List(Value("float32"), length=state_dim),
    "action": Array2D((config.model.chunk, action_dim), "float32"),
    "is_pad": List(Value("bool"), length=config.model.chunk),
  })
  env.close()
  Dataset.from_dict(columns, features=features).save_to_disk(task.dataset)
  print(f"{len(columns['step'])} rows from {len(set(columns['episode']))} episodes -> {task.dataset}")


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])
