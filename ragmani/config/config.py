"""Loads an experiment yaml from this folder into a plain dict.

Classes own their parameters and take their section as keyword arguments, so this module knows only about paths.
"""
from pathlib import Path
from types import SimpleNamespace

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
ROOT = CONFIG_DIR.parent.parent  # act/: relative paths resolve here, whichever directory a script launches from


RENAMED = {  # keys older runs saved under another name, with their values' new spellings
  ("model", "compressed"): (("model", "feature_size"), {}),
  ("model", "draw"): (("model", "keep_sampling"), {"batch": "per_batch", "row": "per_row"}),
  ("model", "score_bias"): (("model", "search_scores_in_attention"), {}),
  ("train", "truth_coin"): (("train", "truth_sampling"), {"batch": "per_batch", "row": "per_row"}),
}


def merge(base, over):
  """base with over's values laid on top, section by section."""
  return {**base, **{key: merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value for key, value in over.items()}}


def load(name, overrides=(), defaults=None):
  """Read config/<name>.yaml (or a path to a yaml), apply "section.key=value" overrides and resolve paths.

  defaults names a config whose values fill keys the yaml lacks, such as a run saved before those keys existed.
  """
  path = Path(name) if str(name).endswith((".yaml", ".yml")) else CONFIG_DIR / f"{name}.yaml"
  config = yaml.safe_load(path.read_text())
  for (section, key), ((new_section, new_key), values) in RENAMED.items():
    if key in config.get(section, {}): value = config[section].pop(key); config[new_section][new_key] = values.get(value, value)
  if defaults: config = merge(yaml.safe_load((CONFIG_DIR / f"{defaults}.yaml").read_text()), config)

  for override in overrides:
    key, separator, value = override.partition("=")
    if not separator: raise ValueError(f"override {override!r} is not key=value")
    *parents, leaf = key.split(".")
    section = config
    for parent in parents: section = section[parent]
    if leaf not in section: raise KeyError(f"override {key!r} names nothing in {path.name}")
    section[leaf] = float(value) if isinstance(section[leaf], float) else yaml.safe_load(value)  # PyYAML reads 3e-4 as a string

  config["paths"] = {key: (ROOT / value).resolve() for key, value in config.get("paths", {}).items()}  # an absolute value stays as is
  config.setdefault("name", path.stem)
  return Section.build(config)


class Section(SimpleNamespace):
  """A config section read by attribute (config.train.lr) or unpacked as kwargs (**config.model)."""

  def keys(self): return vars(self).keys()

  def __getitem__(self, key): return getattr(self, key)

  def to_dict(self):
    """Nested dicts of plain values, paths as strings; named as transformers expects of a model's config, for its W&B logging."""
    return {key: value.to_dict() if isinstance(value, Section) else str(value) if isinstance(value, Path) else value for key, value in vars(self).items()}

  @classmethod
  def build(cls, value):
    """Turn nested dicts into Sections, leaving every other value as is."""
    if isinstance(value, dict): return cls(**{key: cls.build(item) for key, item in value.items()})
    return value
