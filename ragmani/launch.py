"""Sends one training run to the W&B Launch queue that runs it as a Job on Nautilus.

Run from act/ as `uv run --env-file .env python -m ragmani.launch <run> [section.key=value ...]`; an agent watching the queue starts it.
"""
import sys

from wandb.sdk.launch import launch_add

ENTITY = "ragmani"
PROJECT = "ragmani"
QUEUE = "ragmani-nautilus"
JOB = f"{ENTITY}/{PROJECT}/ragmani-train:latest"  # the newest commit registered with `wandb job create`


def main(name, overrides):
  launch_add(job=JOB, queue_name=QUEUE, entity=ENTITY, project=PROJECT, name=name, config={"overrides": {"args": overrides}})
  print(f"queued {name} on {QUEUE} with {len(overrides)} overrides")


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])
