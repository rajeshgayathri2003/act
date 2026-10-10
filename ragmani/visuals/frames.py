"""One frame from the env, at its current state or any stored one, with optional text over it.

Returns an (H, W, 3) uint8 array, so it feeds video.write directly.
"""
import cv2
import numpy as np


def render(env, state=None, **text):
  """The camera frame, after setting state (e.g. a demo's episode.state(step)) when given, with one 'name: value' line per keyword.

  env is unwrapped and reset at least once, as set_state_dict needs.
  """
  if state is not None: env.set_state_dict(state)
  frame = env.render()
  frame = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
  frame = np.ascontiguousarray(frame.reshape(frame.shape[-3:])[..., :3], dtype=np.uint8)  # drop the batch axis and any alpha

  for line, (name, value) in enumerate(text.items()):
    value = np.asarray(value.cpu() if hasattr(value, "cpu") else value)
    words = f"{name}: {np.array2string(value, precision=3, suppress_small=True) if value.ndim else (f'{value:.3f}' if value.dtype.kind == 'f' else value)}"
    origin = (8, 20 + 20 * line)
    cv2.putText(frame, words, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)  # dark outline so the text reads on any background
    cv2.putText(frame, words, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  return frame
