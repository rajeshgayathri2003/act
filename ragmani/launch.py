"""Sends one training run to the W&B Launch queue that runs it as a Job on Nautilus, on code pushed to GitHub.

Run from act/ as `uv run --env-file .env python -m ragmani.launch <run> [section.key=value ...] [--commit <ref>]`; an agent watching the queue starts it.
"""
import argparse
import subprocess
import sys

from wandb.sdk.launch import launch_add

from ragmani.config import load

ENTITY = "ragmani"
PROJECT = "ragmani"
QUEUE = "ragmani-nautilus"
JOB = "ragmani-train"
REPOSITORY = "https://github.com/rajeshgayathri2003/act.git"
IMAGE = "dhbhatt/maniskill-experiments:latest"


def git(*arguments, allowed=(0,)):
  """git's output, exiting with its error message on any return code outside allowed."""
  result = subprocess.run(["git", *arguments], capture_output=True, text=True)
  if result.returncode not in allowed: raise SystemExit(f"git {' '.join(arguments)}: {result.stderr.strip()}")
  return result


def register(reference):
  """The job version for a pushed commit, aliased by its short hash; W&B reuses the version when the commit is already registered."""
  commit = git("rev-parse", "--verify", f"{reference}^{{commit}}").stdout.strip()
  git("fetch", "--quiet", "origin")
  if not git("branch", "--remotes", "--contains", commit).stdout.strip(): raise SystemExit(f"{commit[:7]} is on no branch of origin; push it first, since the Job clones from GitHub")

  alias = commit[:7]
  subprocess.run([sys.executable, "-m", "wandb", "job", "create", "git", REPOSITORY, "-e", ENTITY, "-p", PROJECT, "-n", JOB, "-a", alias, "-g", commit, "-E", "bash ragmani/nautilus.sh", "-B", IMAGE], check=True)
  return f"{ENTITY}/{PROJECT}/{JOB}:{alias}"


def main(arguments):
  load(arguments.config, arguments.overrides)  # a malformed or unknown override fails here, not minutes into the pod's setup
  job = register(arguments.commit)
  launch_add(job=job, queue_name=QUEUE, entity=ENTITY, project=PROJECT, name=arguments.run, run_id=arguments.run_id, config={"overrides": {"args": [arguments.config, *arguments.overrides]}})
  print(f"queued {arguments.run} on {QUEUE}: {job} with {len(arguments.overrides)} overrides")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("run", help="the run's name: its W&B run and folder under /marco-west/runs")
  parser.add_argument("overrides", nargs="*", help="section.key=value on top of the config")
  parser.add_argument("--config", default="push_cube", help="the config in ragmani/config to train")
  parser.add_argument("--commit", default="HEAD", help="the commit or branch to run; it must be pushed to a branch of origin")
  parser.add_argument("--run-id", help="an existing W&B run to continue; its folder on the volume must hold the matching wandb_id")
  main(parser.parse_intermixed_args())  # --commit may come before the overrides
