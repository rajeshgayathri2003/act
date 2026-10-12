"""Frames from the env, at its current state or any stored one, and text written over a frame.

Frames are (H, W, 3) uint8 arrays, so they feed video.write directly.
"""
import cv2
import numpy as np


def render(env, state=None):
  """Every parallel env's camera frame as (envs, H, W, 3), after setting state (e.g. a demo's episode.state(step)) when given.

  env is unwrapped and reset at least once, as set_state_dict needs.
  """
  if state is not None: env.set_state_dict(state)
  images = env.render()
  images = images.cpu().numpy() if hasattr(images, "cpu") else np.asarray(images)
  return np.ascontiguousarray(images.reshape(-1, *images.shape[-3:])[..., :3], dtype=np.uint8)  # a batch axis always, no alpha


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
