"""Opens an encoded dataset for training, retrieval or eval, whichever encoder wrote it.

Callers use data.load and never see the old layout; legacy.convert rebuilds it into this one.
"""
from datasets import load_from_disk

from ragmani.data.legacy import convert


def load(path):
  """An encoded dataset in ragmani's layout, converted when an old encoder wrote it."""
  dataset = load_from_disk(str(path))
  try: dataset.select_columns(["episode", "step", "is_pad"])
  except ValueError: return convert(dataset)
  return dataset
