"""Reads datasets written by the old encoders (ijepa_encoder.py, data_encoder*.py) into ragmani's row layout.

They keep a leading axis of 1 on image_emb, state and next_state and store no episode, step or is_pad, so those are rebuilt here.
"""
import numpy as np
from datasets import Array2D, Dataset, Features, List, Value


def convert(dataset):
  """Old rows to ragmani rows; a row starts an episode where its state is not the previous row's next_state."""
  rows = dataset.with_format("numpy")[:]
  count = len(rows["action"])
  image_emb, state, next_state = (np.asarray(rows[key], np.float32).reshape(count, -1) for key in ("image_emb", "state", "next_state"))
  action = np.asarray(rows["action"], np.float32)

  starts = np.r_[True, (state[1:] != next_state[:-1]).any(axis=1)]
  episode = np.cumsum(starts) - 1
  step = np.arange(count) - np.flatnonzero(starts)[episode]
  is_pad = np.flip(np.maximum.accumulate(np.flip((action != 0).any(axis=-1), axis=-1), axis=-1), axis=-1) == 0  # trailing all-zero rows, as rag_model.py's data_collator
  chunk, action_dim = action.shape[1:]

  features = Features({
    "episode": Value("int32"),
    "step": Value("int32"),
    "image_emb": List(Value("float32"), length=image_emb.shape[1]),
    "state": List(Value("float32"), length=state.shape[1]),
    "next_state": List(Value("float32"), length=state.shape[1]),
    "action": Array2D((chunk, action_dim), "float32"),
    "is_pad": List(Value("bool"), length=chunk),
  })
  return Dataset.from_dict({"episode": episode, "step": step, "image_emb": image_emb, "state": state, "next_state": next_state, "action": action, "is_pad": is_pad}, features=features)
