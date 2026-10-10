"""Loads an experiment yaml from this folder into a plain dict.

Classes own their parameters and take their section as keyword arguments, so this module knows only about paths.
"""
from pathlib import Path
from types import SimpleNamespace

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
ROOT = CONFIG_DIR.parent.parent  # act/: relative paths resolve here, whichever directory a script launches from


def load(name, overrides=()):
  """Read config/<name>.yaml (or a path to a yaml), apply "section.key=value" overrides and resolve paths."""
  path = Path(name) if str(name).endswith((".yaml", ".yml")) else CONFIG_DIR / f"{name}.yaml"
  config = yaml.safe_load(path.read_text())

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
