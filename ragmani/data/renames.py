"""Old parameter names to their current ones, so checkpoints saved before a rename still load."""

RENAMES = {
  "transition_value.": "state_action_next_value.",
}


def rename(weights):
  """The checkpoint's weights with every old name prefix swapped for its current one."""
  for old, new in RENAMES.items():
    weights = {new + key[len(old):] if key.startswith(old) else key: value for key, value in weights.items()}
  return weights
