"""Frames from the env (its current state, any stored one, or a demo's), text written on them, and mp4s of them, alone or tiled.

Frames are (H, W, 3) uint8 arrays, so render's output feeds write directly.
"""
import math
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np

from ragmani.data.trajectories import pick, read


def render(env, camera, state=None):
  """Every parallel env's frame from camera as (envs, H, W, 3), after setting state (e.g. a demo's episode.state(step)) when given.

  env is unwrapped and reset at least once, as set_state_dict needs.
  """
  if state is not None: env.set_state_dict(state)
  images = env.render_rgb_array(camera)
  images = images.cpu().numpy() if hasattr(images, "cpu") else np.asarray(images)
  return np.ascontiguousarray(images.reshape(-1, *images.shape[-3:])[..., :3], dtype=np.uint8)  # a batch axis always, no alpha


def demo(env, path, rows, camera):
  """The frame of each demo (episode, step) in rows, rendered from the state path's h5 recorded there, as many per render as env has copies."""
  with h5py.File(path, "r") as file:
    episodes = {episode: read(file[f"traj_{episode}"]["env_states"]) for episode in {episode for episode, _ in rows}}
  states = [pick(episodes[episode], step) for episode, step in rows]
  images = []
  for start in range(0, len(states), env.num_envs):
    chunk = states[start:start + env.num_envs]
    chunk += [chunk[-1]] * (env.num_envs - len(chunk))  # every copy needs a state; the extras are dropped
    images.extend(render(env, camera, stack(chunk))[:len(states) - start])
  return images


def stack(states):
  """Per-env states as one batched state: each leaf array stacked along a new first axis."""
  return {key: stack([state[key] for state in states]) if isinstance(states[0][key], dict) else np.stack([state[key] for state in states]) for key in states[0]}


def caption(frame, text, size=160):
  """frame resized to size, under a black strip with one line of text."""
  frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
  strip = np.zeros((22, size, 3), dtype=np.uint8)
  cv2.putText(strip, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
  return np.concatenate([strip, frame])


def heading(text, size=160):
  """A black cell one tile wide with text larger than a caption's, to head a column of tiles."""
  cell = np.zeros((32, size, 3), dtype=np.uint8)
  cv2.putText(cell, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
  return cell


def annotate(frame, **text):
  """A copy of frame with one 'name: value' line per keyword."""
  frame = frame.copy()
  for line, (name, value) in enumerate(text.items()):
    value = np.asarray(value.cpu() if hasattr(value, "cpu") else value)
    words = f"{name}: {np.array2string(value, precision=3, suppress_small=True) if value.ndim else (f'{value:.3f}' if value.dtype.kind == 'f' else value)}"
    origin = (8, 20 + 20 * line)
    cv2.putText(frame, words, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)  # dark outline so the text reads on any background
    cv2.putText(frame, words, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  return frame


def write(path, frames, fps=20):
  """Frames to an mp4 at path, creating its folder; the settings test.py used."""
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(path, frames, fps=fps, quality=8)


def grid(episodes, tile=256, successes=None):
  """Episodes (lists of frames) tiled into one sequence of frames; each tile is tile pixels square and an ended episode holds its last frame.

  With successes, one bool per episode, each tile gets a green border for a success and a red one for a failure.
  """
  columns = math.ceil(math.sqrt(len(episodes)))
  rows = math.ceil(len(episodes) / columns)
  length = max(len(frames) for frames in episodes)
  blank = np.zeros((tile, tile, 3), np.uint8)
  tiled = []
  for step in range(length):
    tiles = [cv2.resize(frames[min(step, len(frames) - 1)], (tile, tile), interpolation=cv2.INTER_AREA) for frames in episodes]
    if successes is not None: tiles = [cv2.rectangle(image, (0, 0), (tile - 1, tile - 1), (0, 200, 0) if succeeded else (220, 0, 0), 4) for image, succeeded in zip(tiles, successes)]
    tiles += [blank] * (rows * columns - len(tiles))
    tiled.append(np.concatenate([np.concatenate(tiles[row * columns:(row + 1) * columns], axis=1) for row in range(rows)], axis=0))
  return tiled
