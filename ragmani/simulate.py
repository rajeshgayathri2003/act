"""Runs a trained RagAct in the sim on fresh seeds and records success and video, as Gayathri's test.py did.

Rebuilds the model, retrieval and encoder from the run's own config.yaml, saved by training, and writes beside the run.
Run from act/ as `python -m ragmani.simulate runs/<run> [--checkpoint checkpoint-N] [--episodes N] [--max_steps N]`.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers the ManiSkill envs with gymnasium
import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import TrainerCallback

from ragmani import data
from ragmani.config import load
from ragmani.data.retriever import Retriever
from ragmani.data.trajectories import Trajectories
from ragmani.models import ENCODERS, RagAct
from ragmani.visuals import frames, video

EPISODES = 10
MAX_STEPS = 400   # test.py's default; demos run 61-128 steps, past the env's own 50-step limit
SEED = 1          # draws the episode seeds, as test.py
ANNOTATE = True   # step and action written on each video frame
VIDEOS = 3        # episode videos uploaded to W&B besides the grid; all stay on disk


def environment(config):
  """The task's env under the controller the demos' actions were recorded with, unwrapped for set_state_dict; episode enforces the step cap."""
  return gym.make(config.env_id, num_envs=1, obs_mode="state_dict", control_mode=Trajectories(config.paths.traj).control_mode, render_mode="rgb_array").unwrapped


@torch.no_grad()
def episode(model, retriever, encoder, env, seed, max_steps=MAX_STEPS, annotate=ANNOTATE):
  """One episode from env.reset(seed) until success, termination or max_steps: its frames, whether it succeeded, and per-step rewards."""
  device = next(model.parameters()).device
  observation, _ = env.reset(seed=[seed])
  recorded, rewards, done = [], [], False
  while not done:
    image = encoder.embed([frames.render(env)]).to(device)
    chunk, _ = model(image, observation["agent"]["qpos"].to(device).float().reshape(1, -1), retriever)
    action = chunk[0, 0]  # first step of the predicted chunk
    recorded.append(frames.render(env, step=len(recorded), action=action) if annotate else frames.render(env))
    observation, reward, terminated, truncated, info = env.step(action.cpu()[None])
    rewards.append(float(reward))
    done = bool(terminated or truncated or len(recorded) >= max_steps)
  return recorded, bool(info["success"]), rewards


class Callback(TrainerCallback):
  """For the Trainer: every few epochs, one full episode on main's first seed, logged as video and success at that step."""

  def __init__(self, config, retriever, every):
    self.config, self.retriever, self.every = config, retriever, every
    self.encoder = ENCODERS[config.encoder.name](config.encoder, retriever.rows["image_emb"].device, config.paths.cache)  # the encoder the dataset was built with
    self.env = environment(config)
    self.seed = int(np.random.default_rng(SEED).integers(0, 1_000_000))  # main's episode 0, so the two videos compare

  def on_epoch_end(self, args, state, control, model=None, **kwargs):
    if round(state.epoch) % self.every: return
    model.eval()
    recorded, success, _ = episode(model, self.retriever, self.encoder, self.env, self.seed)
    model.train()
    path = self.config.paths.run_dir / "simulate" / "training" / f"epoch_{round(state.epoch):04d}_{'success' if success else 'fail'}.mp4"
    video.write(path, recorded)
    wandb.log({
      "train/global_step": state.global_step, "train/epoch": state.epoch,
      "simulate/training_success": float(success), "simulate/training_steps": len(recorded),
      "simulate/training_episode": wandb.Video(str(path), caption=f"epoch {round(state.epoch)}: {path.stem}"),
    })


def main(arguments):
  run_dir = Path(arguments.run_dir).resolve()
  config = load(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else load(arguments.config, [f"paths.run_dir={run_dir}"])
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  checkpoint = run_dir / arguments.checkpoint if arguments.checkpoint else max(run_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[1]), default=run_dir / "final")
  output = run_dir / "simulate" / checkpoint.name
  state = json.loads((checkpoint / "trainer_state.json").read_text()) if (checkpoint / "trainer_state.json").exists() else {}
  trained = {"train/global_step": state.get("global_step"), "train/epoch": state.get("epoch")}  # the Trainer's own keys, so simulation points sit on the loss curves' axes
  print(f"Loading checkpoint: {checkpoint} (step {trained['train/global_step']}, epoch {trained['train/epoch']})")

  retriever = Retriever(data.load(config.paths.dataset), config.retrieval, device)
  model = RagAct.load(checkpoint, config.model, image_dim=retriever.rows["image_emb"].shape[-1]).to(device).eval()
  encoder = ENCODERS[config.encoder.name](config.encoder, device, config.paths.cache)
  env = environment(config)

  generator = np.random.default_rng(SEED)
  results, episodes, videos = [], [], []
  for index in tqdm(range(arguments.episodes)):
    seed = int(generator.integers(0, 1_000_000))
    recorded, success, rewards = episode(model, retriever, encoder, env, seed, arguments.max_steps)
    videos.append(output / f"episode_{index:02d}_seed_{seed}_{'success' if success else 'fail'}.mp4")
    video.write(videos[-1], recorded)
    episodes.append(recorded)
    results.append({"episode": index, "seed": seed, "success": success, "steps": len(recorded), "mean_reward": float(np.mean(rewards))})
    print(f"Episode {index}: Total Reward {results[-1]['mean_reward']:.4f} Success {success} Timestep {len(recorded)}")

  video.write(output / "grid.mp4", video.grid(episodes))
  success_rate = sum(result["success"] for result in results) / len(results)
  (output / "summary.json").write_text(json.dumps({"checkpoint": str(checkpoint), "step": trained["train/global_step"], "epoch": trained["train/epoch"], "success_rate": success_rate, "episodes": results}, indent=2))
  print(f"Success rate: {sum(result['success'] for result in results)}/{len(results)} -> {output}")

  id_file = run_dir / "wandb_id"  # written by training, so simulation lands in the run that trained the checkpoint
  with wandb.init(entity=config.wandb.entity, project=config.wandb.project, mode=config.wandb.mode, dir=run_dir, name=run_dir.name, id=id_file.read_text().strip() if id_file.exists() else None, resume="allow") as run:
    id_file.write_text(run.id)
    run.define_metric("simulate/success_rate", summary="max")  # the run's best checkpoint score, for sorting runs
    run.log({
      **{key: value for key, value in trained.items() if value is not None},  # absent when the checkpoint has no trainer_state.json
      "simulate/success_rate": success_rate,
      "simulate/max_steps": arguments.max_steps,
      "simulate/episodes": wandb.Table(columns=["checkpoint", "step", "epoch", *results[0]], data=[[checkpoint.name, trained["train/global_step"], trained["train/epoch"], *result.values()] for result in results]),
      "simulate/grid": wandb.Video(str(output / "grid.mp4"), caption=checkpoint.name),
      **{f"simulate/episode_{index:02d}": wandb.Video(str(path), caption=f"{checkpoint.name}: {path.stem}") for index, path in enumerate(videos[:VIDEOS])},  # stable keys, so a panel steps through checkpoints
    })


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("run_dir", help="a training run folder, e.g. runs/push_cube")
  parser.add_argument("--checkpoint", help="a checkpoint-* folder or final; defaults to the latest checkpoint")
  parser.add_argument("--config", default="push_cube", help="config for runs with no config.yaml of their own")
  parser.add_argument("--episodes", type=int, default=EPISODES)
  parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
  main(parser.parse_args())
