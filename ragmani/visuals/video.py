"""Writes frame sequences to mp4, alone or tiled so several episodes play side by side."""
import math
from pathlib import Path

import cv2
import imageio
import numpy as np


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
