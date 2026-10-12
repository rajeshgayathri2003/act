"""Sends one training run to the W&B Launch queue that runs it as a Job on Nautilus, on code pushed to marco-dev.

Run from act/ as `uv run --env-file .env python -m ragmani.launch <run> [section.key=value ...] [--commit <ref>]`; an agent watching the queue starts it.
Run from act/ as `uv run --env-file .env python -m ragmani.launch <run> [section.key=value ...] [--commit <ref>]`; an agent watching the queue starts it.
"""
import argparse
import subprocess
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
BRANCH = "marco-dev"
IMAGE = "dhbhatt/maniskill-experiments:latest"


def git(*arguments):
  return subprocess.run(["git", *arguments], check=True, capture_output=True, text=True).stdout.strip()


def register(reference):
  """The job version for a pushed commit, aliased by its short hash; W&B reuses the version when the commit is already registered."""
  commit = git("rev-parse", "--verify", f"{reference}^{{commit}}")
  git("fetch", "--quiet", "origin", BRANCH)
  if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "FETCH_HEAD"]).returncode: raise SystemExit(f"{commit[:7]} is not on origin/{BRANCH}; push it first, since the Job clones from GitHub")

  alias = commit[:7]
  subprocess.run([sys.executable, "-m", "wandb", "job", "create", "git", REPOSITORY, "-e", ENTITY, "-p", PROJECT, "-n", JOB, "-a", alias, "-g", commit, "-E", "bash ragmani/nautilus.sh", "-B", IMAGE], check=True)
  return f"{ENTITY}/{PROJECT}/{JOB}:{alias}"


def main(arguments):
  job = register(arguments.commit)
  launch_add(job=job, queue_name=QUEUE, entity=ENTITY, project=PROJECT, name=arguments.run, config={"overrides": {"args": arguments.overrides}})
  print(f"queued {arguments.run} on {QUEUE}: {job} with {len(arguments.overrides)} overrides")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("run", help="the run's name: its W&B run and folder under /marco-west/runs")
  parser.add_argument("overrides", nargs="*", help="section.key=value on top of push_cube.yaml")
  parser.add_argument("--commit", default="HEAD", help="the commit or branch to run; it must be pushed to marco-dev")
  main(parser.parse_args())
