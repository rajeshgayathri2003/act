"""Runs a trained RagAct in the sim on fresh layouts and records success and video, as Gayathri's test.py does.

Every episode runs at once in ManiSkill's parallel envs, rebuilt from the run's own config.yaml.
Run from act/ as `uv run python -m ragmani.simulate runs/<run> [--checkpoint checkpoint-N] [--episodes N] [--max_steps N]`.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers the ManiSkill envs with gymnasium
import numpy as np
import torch
import wandb
from safetensors.torch import save_model
from transformers import TrainerCallback

from ragmani import data
from ragmani.config import load
from ragmani.data.retriever import Retriever
from ragmani.data.trajectories import Trajectories
from ragmani.models import ENCODERS, RagAct
from ragmani.visuals import frames, video

EPISODES = 20
MAX_STEPS = 128   # the longest demo; test.py allowed 400
SEED = 42         # picks the episodes' starting layouts; test.py used 1
CHECK_SEED = 7    # picks the layouts scored during training; not SEED, so choosing best_success never sees the layouts simulate reports
ANNOTATE = True   # step and action written on each video frame
VIDEOS = 3        # episode videos uploaded to W&B besides the grid; all stay on disk
ENSEMBLE_DECAY = 0.01  # ACT's k in exp(-k * i): older chunks count slightly more


def ensemble(chunks, step):
  """ACT's temporal ensembling for every env at once: each chunk predicted for this step votes, weighted exp(-ENSEMBLE_DECAY * i) from the oldest."""
  actions = torch.stack([chunk[:, step - start] for start, chunk in chunks if step - start < chunk.shape[1]])  # (votes, envs, action_dim)
  weights = torch.exp(-ENSEMBLE_DECAY * torch.arange(len(actions), device=actions.device))
  return (weights[:, None, None] * actions).sum(dim=0) / weights.sum()


EXECUTIONS = {  # which action each env runs at a step, from the (start step, chunks) pairs predicted so far
  "ensemble": ensemble,
  "first": lambda chunks, step: chunks[-1][1][:, 0],  # the newest chunk's first action, discarding the rest
}


def environment(config, count=1):
  """count copies of the task's env stepped together, on GPU physics when more than one, under the demos' controller; unwrapped for set_state_dict."""
  return gym.make(config.env_id, num_envs=count, sim_backend="physx_cuda" if count > 1 else "auto", obs_mode="state_dict", control_mode=Trajectories(config.paths.traj).control_mode, render_mode="rgb_array").unwrapped


@torch.no_grad()
def episodes(model, retriever, encoder, env, seeds, max_steps=MAX_STEPS, execution="ensemble", keep=None, annotate=ANNOTATE):
  """One episode per seed in env's parallel copies, each until success or max_steps: per episode its frames, success and per-step rewards.

  Frames are kept only for the episodes indexed in keep (all when None).
  """
  device = next(model.parameters()).device
  observation, _ = env.reset(seed=seeds)
  recorded, rewards, chunks = [[] for _ in seeds], [[] for _ in seeds], []
  succeeded, done = torch.zeros(len(seeds), dtype=torch.bool), torch.zeros(len(seeds), dtype=torch.bool)
  for step in range(max_steps):
    images = frames.render(env)
    embeddings, qpos = encoder.embed(images).to(device), observation["agent"]["qpos"].to(device).float()
    chunk, _, _ = model(embeddings, qpos, retriever)
    chunk = chunk * model.action.std + model.action.mean  # back to the controller's units
    chunks = [*chunks, (step, chunk)][-chunk.shape[1]:]  # only chunks that still cover this step
    action = EXECUTIONS[execution](chunks, step)
    observation, reward, terminated, _, info = env.step(action.to(env.device))  # the unwrapped env never truncates; max_steps does

    for index in (~done).nonzero().flatten().tolist():
      rewards[index].append(float(reward.reshape(-1)[index]))
      if keep is not None and index not in keep: continue
      recorded[index].append(frames.annotate(images[index], step=step, action=action[index]) if annotate else images[index])

    success = info["success"].reshape(-1).cpu()
    succeeded |= success & ~done
    done |= success | terminated.reshape(-1).cpu()
    if done.all(): break
  return [(recorded[index], bool(succeeded[index]), rewards[index]) for index in range(len(seeds))]


class Callback(TrainerCallback):
  """For the Trainer: every few epochs, a video of one fixed layout and the success rate over other fixed layouts, kept as best_success when highest."""

  def __init__(self, config, retriever, every, episodes):
    self.config, self.retriever, self.every = config, retriever, every
    score = config.paths.run_dir / "best_success" / "success_rate.json"
    self.highest = json.loads(score.read_text())["success_rate"] if config.train.resume and score.exists() else -1.0  # a resumed run keeps the rate best_success/ was saved at
    self.encoder = ENCODERS[config.encoder.name](config.encoder, retriever.rows["image_emb"].device, config.paths.cache)  # the encoder the dataset was built with
    video_seed = int(np.random.default_rng(SEED).integers(0, 1_000_000))  # main's episode 0, so the two videos compare
    scored_seeds = np.random.default_rng(CHECK_SEED).integers(0, 1_000_000, episodes).tolist()
    self.seeds = [video_seed, *scored_seeds]
    self.env = environment(config, len(self.seeds))

  def on_epoch_end(self, args, state, control, model=None, **kwargs):
    if round(state.epoch) % self.every: return
    model.eval()
    (recorded, success, _), *others = episodes(model, self.retriever, self.encoder, self.env, self.seeds, keep={0})
    scored = [succeeded for _, succeeded, _ in others]
    model.train()
    if scored and np.mean(scored) >= self.highest:  # a tie keeps the later, longer-trained model
      self.highest = float(np.mean(scored))
      (self.config.paths.run_dir / "best_success").mkdir(parents=True, exist_ok=True)
      save_model(model, str(self.config.paths.run_dir / "best_success" / "model.safetensors"))
      state.save_to_json(str(self.config.paths.run_dir / "best_success" / "trainer_state.json"))
      (self.config.paths.run_dir / "best_success" / "success_rate.json").write_text(json.dumps({"success_rate": self.highest, "epoch": state.epoch}))
    path = self.config.paths.run_dir / "simulate" / "training" / f"epoch_{round(state.epoch):04d}_{'success' if success else 'fail'}.mp4"
    video.write(path, recorded)
    wandb.log({
      "train/global_step": state.global_step, "train/epoch": state.epoch,
      "simulate/training_success": float(success), "simulate/training_steps": len(recorded),
      **({"simulate/training_success_rate": float(np.mean(scored))} if scored else {}),
      "simulate/training_episode": wandb.Video(str(path), caption=f"epoch {round(state.epoch)}: {path.stem}"),
    })


def main(arguments):
  run_dir = Path(arguments.run_dir).resolve()
  config = load(run_dir / "config.yaml", defaults=arguments.config) if (run_dir / "config.yaml").exists() else load(arguments.config, [f"paths.run_dir={run_dir}"])  # older runs lack newer keys
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  checkpoint = run_dir / arguments.checkpoint if arguments.checkpoint else max(run_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[1]), default=run_dir / "final")
  output = run_dir / "simulate" / f"{checkpoint.name}-{arguments.execution}"
  state = json.loads((checkpoint / "trainer_state.json").read_text()) if (checkpoint / "trainer_state.json").exists() else {}
  trained = {"train/global_step": state.get("global_step"), "train/epoch": state.get("epoch")}  # the Trainer's own keys, so simulation points sit on the loss curves' axes
  print(f"Loading checkpoint: {checkpoint} (step {trained['train/global_step']}, epoch {trained['train/epoch']})")

  retriever = Retriever(data.load(config.paths.dataset), config.retrieval, device)
  model = RagAct.load(checkpoint, config.model, image_dim=retriever.rows["image_emb"].shape[-1]).to(device).eval()
  encoder = ENCODERS[config.encoder.name](config.encoder, device, config.paths.cache)
  seeds = np.random.default_rng(SEED).integers(0, 1_000_000, arguments.episodes).tolist()
  env = environment(config, len(seeds))
  outcomes = episodes(model, retriever, encoder, env, seeds, arguments.max_steps, arguments.execution)
  env.close()  # training calls main once per checkpoint, and each env holds GPU physics memory

  results, recordings, videos = [], [], []
  for index, (seed, (recorded, success, rewards)) in enumerate(zip(seeds, outcomes)):
    videos.append(output / f"episode_{index:02d}_seed_{seed}_{'success' if success else 'fail'}.mp4")
    video.write(videos[-1], recorded)
    recordings.append(recorded)
    results.append({"episode": index, "seed": seed, "success": success, "steps": len(recorded), "mean_reward": float(np.mean(rewards))})
    print(f"Episode {index}: Total Reward {results[-1]['mean_reward']:.4f} Success {success} Timestep {len(recorded)}")

  video.write(output / "grid.mp4", video.grid(recordings))
  successes = sum(result["success"] for result in results)
  success_rate = successes / len(results)
  (output / "summary.json").write_text(json.dumps({"checkpoint": str(checkpoint), "step": trained["train/global_step"], "epoch": trained["train/epoch"], "execution": arguments.execution, "success_rate": success_rate, "episodes": results}, indent=2))
  print(f"Success rate: {successes}/{len(results)} -> {output}")

  id_file = run_dir / "wandb_id"  # written by training, so simulation lands in the run that trained the checkpoint
  with wandb.init(entity=config.wandb.entity, project=config.wandb.project, mode=config.wandb.mode, dir=run_dir.parent, name=run_dir.name, id=id_file.read_text().strip() if id_file.exists() else None, resume="allow") as run:
    id_file.write_text(run.id)
    run.define_metric("simulate/success_rate", summary="max")  # the run's best checkpoint score, for sorting runs
    run.log({
      **{key: value for key, value in trained.items() if value is not None},  # absent when the checkpoint has no trainer_state.json
      "simulate/success_rate": success_rate,
      "simulate/max_steps": arguments.max_steps,
      "simulate/execution": arguments.execution,
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
  parser.add_argument("--execution", choices=EXECUTIONS, default="ensemble", help="how a step's action comes from the predicted chunks")
  main(parser.parse_args())
