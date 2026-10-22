"""Runs ragmani on Nautilus through your W&B Launch queue ragmani-$RAGMANI_USER: set the queue up, run its agent, queue runs, and start one inside its Job.

From act/: `python -m ragmani.launch <run> --preset <name> [--commit <ref>] [--smoke] [--section.key value ...]` queues a run; `setup`, `agent` and `job` are the other commands.
Any flag a run does not know is an override, checked against the preset before anything is queued.
"""
import argparse
import os
import string
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from wandb.sdk.launch import launch_add

import wandb
from ragmani.config import cli

ENTITY = "ragmani"
PROJECT = "ragmani"
JOB = "ragmani-train"
REPOSITORY = "https://github.com/rajeshgayathri2003/act.git"
IMAGE = "gitlab-registry.nrp-nautilus.io/ragmani/ragmani@sha256:0c1dd46f8d104ec5ec3da4fbbe570ec97982e0d4f2f468848a63a699d1113b21"  # docker/Dockerfile for amd64 and arm64, pinned so the image cannot change under a run
ENTRYPOINT = "uv run --frozen python -m ragmani.launch job"  # the image has uv and the environment; Launch appends the run's preset and overrides
SMOKE = ["--train.epochs", "1", "--train.simulate_epochs", "1", "--train.simulate_episodes", "2", "--train.simulate_after.episodes", "0", "--train.no_resume", "--wandb.mode", "offline"]


def setting(name):
  """A per-person setting from .env."""
  if not (value := os.environ.get(name)): raise SystemExit(f"set {name} in .env; the README's Launching section lists what each person sets")
  return value


def queue():
  """Your queue, named after RAGMANI_USER."""
  return f"ragmani-{setting('RAGMANI_USER')}"


def setup():
  """Creates or updates your queue from queue.yaml, with your user, volume and W&B secret filled in."""
  template = string.Template((Path(__file__).parent / "queue.yaml").read_text())
  config = yaml.safe_load(template.safe_substitute(user=setting("RAGMANI_USER"), volume=setting("RAGMANI_VOLUME"), wandb_secret=setting("RAGMANI_WANDB_SECRET")))  # ${run_id} stays for Launch
  wandb.Api().upsert_run_queue(queue(), config, "kubernetes", entity=ENTITY)
  print(f"{ENTITY}/{queue()} is set from queue.yaml")


def agent(*args):
  """wandb launch-agent on your queue, with a Kubernetes token that renews so its Job watcher keeps seeing Jobs finish.

  The watcher keeps one client for the agent's life, and kubernetes_asyncio runs the kubeconfig's exec login (NRP's oidc-login, 30-minute tokens) only once. Needs `uv sync --group launch`.
  """
  from kubernetes_asyncio.config.kube_config import KubeConfigLoader
  from wandb.cli.cli import cli as wandb_cli

  set_config = KubeConfigLoader._set_config

  def renewing(loader, configuration):
    set_config(loader, configuration)
    if not loader._user or "exec" not in loader._user: return

    async def renew(configuration):  # runs before each request
      if await loader.load_from_exec_plugin(): set_config(loader, configuration)  # it returns at once while the token is still valid
    configuration.refresh_api_key_hook = renew

  KubeConfigLoader._set_config = renewing  # private API: recheck after a kubernetes_asyncio upgrade
  wandb_cli(["launch-agent", "-e", ENTITY, "-q", queue(), *args])


def job(preset, *overrides):
  """Inside the Job: trains the run in its folder under RUNS_DIR; a folder another run holds gets this run's id appended, and its own folder resumes."""
  runs, name, run_id = Path(os.environ["RUNS_DIR"]), os.environ["WANDB_NAME"], os.environ["WANDB_RUN_ID"]
  run_dir = runs / name
  if run_dir.exists() and (run_dir / "wandb_id").exists() and (run_dir / "wandb_id").read_text().strip() != run_id: run_dir = runs / f"{name}-{run_id}"
  os.execv(sys.executable, [sys.executable, "-m", "ragmani.train", preset, "--run_dir", str(run_dir), *overrides])


def git(*args, allowed=(0,)):
  """git's output, exiting with its error message on any return code outside allowed."""
  result = subprocess.run(["git", *args], capture_output=True, text=True)
  if result.returncode not in allowed: raise SystemExit(f"git {' '.join(args)}: {result.stderr.strip()}")
  return result


def register(commit):
  """The job version for a pushed commit, aliased by its short hash; W&B reuses the version when the commit is already registered."""
  git("fetch", "--quiet", "origin")
  if not git("branch", "--remotes", "--contains", commit).stdout.strip(): raise SystemExit(f"{commit[:7]} is on no branch of origin; push it first, since the Job clones from GitHub")
  alias = commit[:7]
  subprocess.run([sys.executable, "-m", "wandb", "job", "create", "git", REPOSITORY, "-e", ENTITY, "-p", PROJECT, "-n", JOB, "-a", alias, "-g", commit, "-E", ENTRYPOINT, "-B", IMAGE], check=True)
  return f"{ENTITY}/{PROJECT}/{JOB}:{alias}"


def run(*args):
  """Queues one run, after checking its overrides and, with --smoke, training them here for an epoch."""
  parser = argparse.ArgumentParser(description="queue one run on your queue")
  parser.add_argument("run", help="the run's name: its W&B run, Job name and folder under RUNS_DIR; lowercase, digits and dashes")
  parser.add_argument("--preset", default="push_cube", help="the preset in ragmani/config.py to train")
  parser.add_argument("--commit", default="HEAD", help="the commit or branch to run; it must be pushed to a branch of origin")
  parser.add_argument("--run-id", help="the W&B run id; defaults to the run's name, and an existing run's id continues it")
  parser.add_argument("--smoke", action="store_true", help="first train the same flags here for one epoch, from this checkout, which must be the commit")
  arguments, overrides = parser.parse_known_args(args)  # the flags it does not know are the overrides, in their order
  train_arguments = [arguments.preset, *overrides]
  cli(train_arguments)  # a malformed or unknown override fails here, not minutes into the pod's setup
  commit = git("rev-parse", "--verify", f"{arguments.commit}^{{commit}}").stdout.strip()

  if arguments.smoke:
    if commit != git("rev-parse", "HEAD").stdout.strip() or git("status", "--porcelain", "--untracked-files=no").stdout.strip(): raise SystemExit("--smoke trains this checkout, so check out the commit, with no uncommitted changes")
    with tempfile.TemporaryDirectory() as folder:
      if subprocess.run([sys.executable, "-m", "ragmani.train", *train_arguments, "--run_dir", str(Path(folder) / "smoke"), *SMOKE]).returncode: raise SystemExit("the smoke run failed; nothing was queued")

  job_version = register(commit)
  launch_add(job=job_version, queue_name=queue(), entity=ENTITY, project=PROJECT, name=arguments.run, run_id=arguments.run_id or arguments.run, config={"overrides": {"args": train_arguments}})
  print(f"queued {arguments.run} on {queue()}: {job_version} with overrides {overrides}")


COMMANDS = {  # the first argument picks one; anything else queues a run
  "setup": setup,
  "agent": agent,
  "job": job,
}


if __name__ == "__main__":
  command, *rest = sys.argv[1:] or [""]
  if command in COMMANDS: COMMANDS[command](*rest)
  else: run(*sys.argv[1:])
