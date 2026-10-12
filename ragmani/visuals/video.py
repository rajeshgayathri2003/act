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


def grid(episodes, tile=256):
  """Episodes (lists of frames) tiled into one sequence of frames; each tile is tile pixels square and an ended episode holds its last frame."""
  columns = math.ceil(math.sqrt(len(episodes)))
  rows = math.ceil(len(episodes) / columns)
  length = max(len(frames) for frames in episodes)
  blank = np.zeros((tile, tile, 3), np.uint8)
  tiled = []
  for step in range(length):
    tiles = [cv2.resize(frames[min(step, len(frames) - 1)], (tile, tile), interpolation=cv2.INTER_AREA) for frames in episodes]
    tiles += [blank] * (rows * columns - len(tiles))
    tiled.append(np.concatenate([np.concatenate(tiles[row * columns:(row + 1) * columns], axis=1) for row in range(rows)], axis=0))
  return tiled
