"""Runs a trained RagAct in the sim on fresh layouts and records success and video, as Gayathri's test.py does.

Every episode runs at once in ManiSkill's parallel envs, with the model rebuilt from the config W&B stored for the run.
Run from act/ as `uv run python -m ragmani.simulate runs/<run> [--checkpoint checkpoint-N] [--episodes N] [--max_steps N]`.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers the ManiSkill envs with gymnasium
import numpy as np
import torch
import wandb
from safetensors.torch import save_model
from transformers import TrainerCallback

from ragmani import data
from ragmani.config import cli, load_run
from ragmani.data.retriever import Retriever
from ragmani.data.trajectories import Trajectories
from ragmani.models import ENCODERS, RagAct
from ragmani.visuals import frames, video

EPISODES = 20
MAX_STEPS = 128   # the longest demo; test.py allowed 400
SEED = 42         # picks the episodes' starting layouts; test.py used 1
CHECK_SEED = 7    # picks the layouts scored during training; not SEED, so choosing best_success never sees the layouts simulate reports
ANNOTATE = True   # step and action written on each video frame
ENSEMBLE_DECAY = 0.01  # ACT's k in exp(-k * i): older chunks count slightly more


def ensemble(chunks, step):
  """ACT's temporal ensembling for every env at once: each chunk predicted for this step votes, weighted exp(-ENSEMBLE_DECAY * i) from the oldest and by how sure the model is the step is a real action.

  Steps past the demo's end get no action target in training, so a step the model marks as padding barely counts.
  """
  votes = [(chunk[:, step - start], real[:, step - start]) for start, chunk, real in chunks if step - start < chunk.shape[1]]
  actions = torch.stack([action for action, _ in votes])  # (votes, envs, action_dim)
  real = torch.stack([real for _, real in votes])  # (votes, envs)
  weights = torch.exp(-ENSEMBLE_DECAY * torch.arange(len(votes), device=actions.device))[:, None] * real
  return (weights[..., None] * actions).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-6)[:, None]


EXECUTIONS = {  # which action each env runs at a step, from the (start step, chunks) pairs predicted so far
  "ensemble": ensemble,
  "first": lambda chunks, step: chunks[-1][1][:, 0],  # the newest chunk's first action, discarding the rest
}


def environment(task, count=1):
  """count copies of the task's env stepped together, on GPU physics when more than one, under the demos' controller; unwrapped for set_state_dict."""
  return gym.make(task.env_id, num_envs=count, sim_backend="physx_cuda" if count > 1 else "auto", obs_mode="state_dict", control_mode=Trajectories(task.traj).control_mode, render_mode="rgb_array").unwrapped


def key(section, tasks, task, name):
  """A W&B key under section (check or sim), with the task's env_id between when the run has several tasks."""
  return f"{section}/{name}" if len(tasks) == 1 else f"{section}/{task.env_id}/{name}"


def share(count, tasks, seed):
  """The task of each of count episodes: an even split over the tasks, shuffled, so any number of tasks costs the same and each gets some."""
  return np.random.default_rng(seed).permutation(np.arange(count) % tasks).tolist()


@torch.no_grad()
def episodes(model, retriever, encoder, envs, seeds, max_steps=MAX_STEPS, execution="ensemble", annotate=ANNOTATE):
  """For each env, one episode per seed in its list, in its parallel copies, each until success or max_steps: per env, per episode its frames, success and per-step rewards.

  Every env steps together, so each step encodes their frames in one batch and predicts in one forward.
  """
  device = next(model.parameters()).device
  observations = [env.reset(seed=env_seeds)[0] for env, env_seeds in zip(envs, seeds)]
  recorded, rewards = [[[] for _ in env_seeds] for env_seeds in seeds], [[[] for _ in env_seeds] for env_seeds in seeds]
  chunks = [[] for _ in envs]  # per env, (start step, actions, real-step probabilities) for each chunk still covering a step
  succeeded, done = [torch.zeros(len(env_seeds), dtype=torch.bool) for env_seeds in seeds], [torch.zeros(len(env_seeds), dtype=torch.bool) for env_seeds in seeds]
  for step in range(max_steps):
    active = [number for number in range(len(envs)) if not done[number].all()]
    if not active: break
    images = [frames.render(envs[number]) for number in active]
    embeddings = encoder.embed(np.concatenate(images)).to(device)
    qpos = torch.cat([observations[number]["agent"]["qpos"].to(device).float() for number in active])
    chunk, pad_logit, _ = model(embeddings, qpos, retriever)
    chunk = model.action.destandardize(chunk)  # back to the controller's units
    real = torch.sigmoid(-pad_logit)  # (envs, chunk_len): how sure the model is each step is a real action, not past the demo's end

    sizes = [len(seeds[number]) for number in active]
    for number, image, part, part_real in zip(active, images, chunk.split(sizes), real.split(sizes)):
      chunks[number] = [*chunks[number], (step, part, part_real)][-part.shape[1]:]  # only chunks that still cover this step
      action = EXECUTIONS[execution](chunks[number], step)
      observations[number], reward, terminated, _, info = envs[number].step(action.to(envs[number].device))  # the unwrapped env never truncates; max_steps does
      for index in (~done[number]).nonzero().flatten().tolist():
        rewards[number][index].append(float(reward.reshape(-1)[index]))
        recorded[number][index].append(frames.annotate(image[index], step=step, action=action[index]) if annotate else image[index])
      success = info["success"].reshape(-1).cpu()
      succeeded[number] |= success & ~done[number]
      done[number] |= success | terminated.reshape(-1).cpu()
  return [[(recorded[number][index], bool(succeeded[number][index]), rewards[number][index]) for index in range(len(seeds[number]))] for number in range(len(envs))]
    images = frames.render(env)
    embeddings, qpos = encoder.embed(images).to(device), observation["agent"]["qpos"].to(device).float()
    chunk, _, _, _ = model(embeddings, qpos, retriever)
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
  """For the Trainer: every few epochs, a check under check/: the success rate over fixed layouts (kept as best_success when highest), their grid, and a video of one more fixed layout.

  Each check also logs retrieval/neighbours: for one fixed held-out frame, the rows a query of its own embedding finds and the rows each search kept.
  """

  def __init__(self, config, retriever, eval_rows, every, episodes):
    self.config, self.retriever, self.every = config, retriever, every
    episode = eval_rows["episode"][:].numpy()
    truth_row = int(np.flatnonzero(episode == episode.min())[(episode == episode.min()).sum() // 2])  # the middle step of the first held-out episode
    truth = eval_rows.with_format("torch", columns=["image_emb", "state", "episode", "step"])[truth_row:truth_row + 1]
    self.truth = {key: value.to(retriever.rows["image_emb"].device) for key, value in truth.items()}
    score = config.run_dir / "best_success" / "success_rate.json"
    self.highest = json.loads(score.read_text())["success_rate"] if config.train.resume and score.exists() else -1.0  # a resumed run keeps the rate best_success/ was saved at
    self.encoder = ENCODERS[config.encoder.name](config.encoder, retriever.rows["image_emb"].device)  # the encoder the dataset was built with
    video_seed = int(np.random.default_rng(SEED).integers(0, 1_000_000))  # main's episode 0, so the two videos compare
    scored_seeds = np.random.default_rng(CHECK_SEED).integers(0, 1_000_000, episodes).tolist()
    tasks = share(len(scored_seeds), len(config.tasks), CHECK_SEED)
    self.seeds = [[video_seed, *(seed for seed, task in zip(scored_seeds, tasks) if task == index)] for index in range(len(config.tasks))]  # each task films the video layout, then scores its share
    self.envs = [environment(task, len(seeds)) for task, seeds in zip(config.tasks, self.seeds)]

  def on_epoch_end(self, args, state, control, model=None, **kwargs):
    if round(state.epoch) % self.every: return
    model.eval()
    outcomes = episodes(model, self.retriever, self.encoder, self.envs, self.seeds)
    neighbours = self.neighbours(model)
    model.train()
    logs, every = {"train/global_step": state.global_step, "train/epoch": state.epoch}, []
    for task, ((recorded, success, _), *others) in zip(self.config.tasks, outcomes):
      scored = [succeeded for _, succeeded, _ in others]
      every += scored
      logs |= self.log(task, round(state.epoch), recorded, success, others, scored)
    if every and np.mean(every) >= self.highest:  # a tie keeps the later, longer-trained model
      self.highest = float(np.mean(every))
      (self.config.run_dir / "best_success").mkdir(parents=True, exist_ok=True)
      save_model(model, str(self.config.run_dir / "best_success" / "model.safetensors"))
      state.save_to_json(str(self.config.run_dir / "best_success" / "trainer_state.json"))
      (self.config.run_dir / "best_success" / "success_rate.json").write_text(json.dumps({"success_rate": self.highest, "epoch": state.epoch}))
    wandb.log({
      **logs,
      **({"check/success_rate": float(np.mean(every))} if every else {}),  # over every scored episode, whichever its task
      "retrieval/neighbours": wandb.Image(neighbours, caption=f"epoch {round(state.epoch)}: truth, then each search's kept rows"),
    })

  def log(self, task, epoch, recorded, success, others, scored):
    """One task's check: its videos on disk and its W&B entries, under check/<env_id>/ when the run has several tasks."""
    folder = "training" if len(self.config.tasks) == 1 else f"training/{task.env_id}"
    path = self.config.run_dir / "simulate" / folder / f"epoch_{epoch:04d}_{'success' if success else 'fail'}.mp4"
    video.write(path, recorded)
    grid_path = path.with_name(f"epoch_{epoch:04d}_grid.mp4")
    video.write(grid_path, video.grid([frames for frames, _, _ in others], tile=160, successes=scored))  # every scored layout, bordered green or red
    tasks = self.config.tasks
    return {
      **({key("check", tasks, task, "success_rate"): float(np.mean(scored))} if scored and len(tasks) > 1 else {}),
      key("check", tasks, task, "video"): wandb.Video(str(path), caption=f"epoch {epoch}: {path.stem}"),
      key("check", tasks, task, "grid"): wandb.Video(str(grid_path), caption=f"epoch {epoch}: {sum(scored)}/{len(scored)} succeeded"),
    }

  @torch.no_grad()
  def neighbours(self, model):
    """One image: columns of the truth, the rows a query of its own embedding finds and the rows each search kept; row k holds each column's k-th nearest."""
    keep, rows = self.config.model.n_keep, self.retriever.rows
    _, _, idx = model(self.truth["image_emb"], self.truth["state"], self.retriever)  # (1, queries*keep), nearest first in eval mode
    truth_idx = self.retriever.search(self.truth["image_emb"][:, None].float(), keep, self.truth["episode"]).flatten()  # the reference the retrieval scores call 1
    columns = [("query", truth_idx)] + [(f"search {search + 1}", idx[0, search * keep:(search + 1) * keep]) for search in range(self.config.model.n_queries)]

    locations = [(int(self.truth["episode"]), int(self.truth["step"]))] + [(int(rows["episode"][row]), int(rows["step"][row])) for _, column in columns for row in column.tolist()]
    images = [None] * len(locations)
    for index, task in enumerate(self.config.tasks):  # each row renders in its own task's env, from its own demos
      mine = [number for number, (episode, _) in enumerate(locations) if data.split(episode)[0] == index]
      for number, image in zip(mine, frames.demo(self.envs[index], task.traj, [(data.split(locations[number][0])[1], locations[number][1]) for number in mine])): images[number] = image
    tiles = [frames.caption(image, f"ep {episode} s {step}") for image, (episode, step) in zip(images, locations)]
    grid = [[tiles[0]] * keep] + [tiles[1 + number * keep:1 + (number + 1) * keep] for number in range(len(columns))]  # one list of keep tiles per column
    headings = np.concatenate([frames.heading(label) for label in ["truth", *(label for label, _ in columns)]], axis=1)
    return np.concatenate([headings, *(np.concatenate([column[rank] for column in grid], axis=1) for rank in range(keep))])


def main(arguments):
  run_dir = Path(arguments.run_dir).resolve()
  id_file = run_dir / "wandb_id"  # written by training, so simulation lands in the run that trained the checkpoint
  config = load_run(id_file.read_text().strip()) if id_file.exists() else cli([arguments.config, "--run_dir", str(run_dir)])
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  checkpoint = run_dir / arguments.checkpoint if arguments.checkpoint else max(run_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[1]), default=run_dir / "final")
  output = run_dir / "simulate" / f"{checkpoint.name}-{arguments.execution}"
  state = json.loads((checkpoint / "trainer_state.json").read_text()) if (checkpoint / "trainer_state.json").exists() else {}
  trained = {"train/global_step": state.get("global_step"), "train/epoch": state.get("epoch")}  # the Trainer's own keys, so simulation points sit on the loss curves' axes
  print(f"Loading checkpoint: {checkpoint} (step {trained['train/global_step']}, epoch {trained['train/epoch']})")

  retriever = Retriever(data.combine(config.tasks), config.retrieval, device)
  config.model.d_image = retriever.rows["image_emb"].shape[-1]
  model = RagAct.load(checkpoint, config.model).to(device).eval()
  encoder = ENCODERS[config.encoder.name](config.encoder, device)
  seeds = np.random.default_rng(SEED).integers(0, 1_000_000, arguments.episodes).tolist()
  tasks = share(len(seeds), len(config.tasks), SEED)
  runs = [(task, [seed for seed, assigned in zip(seeds, tasks) if assigned == index]) for index, task in enumerate(config.tasks)]
  runs = [(task, task_seeds) for task, task_seeds in runs if task_seeds]
  envs = [environment(task, len(task_seeds)) for task, task_seeds in runs]
  outcomes = episodes(model, retriever, encoder, envs, [task_seeds for _, task_seeds in runs], arguments.max_steps, arguments.execution)
  for env in envs: env.close()  # training calls main once per checkpoint, and each env holds GPU physics memory

  logs, every, table = {}, [], []
  for (task, task_seeds), task_outcomes in zip(runs, outcomes):
    folder = output if len(config.tasks) == 1 else output / task.env_id
    results, recordings = [], []
    for index, (seed, (recorded, success, rewards)) in enumerate(zip(task_seeds, task_outcomes)):
      video.write(folder / f"episode_{index:02d}_seed_{seed}_{'success' if success else 'fail'}.mp4", recorded)
      recordings.append(recorded)
      results.append({"episode": index, "seed": seed, "success": success, "steps": len(recorded), "mean_reward": float(np.mean(rewards))})
      print(f"{task.env_id} episode {index}: Total Reward {results[-1]['mean_reward']:.4f} Success {success} Timestep {len(recorded)}")

    video.write(folder / "grid.mp4", video.grid(recordings, successes=[result["success"] for result in results]))
    successes = sum(result["success"] for result in results)
    every += [result["success"] for result in results]
    (folder / "summary.json").write_text(json.dumps({"checkpoint": str(checkpoint), "env_id": task.env_id, "step": trained["train/global_step"], "epoch": trained["train/epoch"], "execution": arguments.execution, "success_rate": successes / len(results), "episodes": results}, indent=2))
    print(f"{task.env_id} success rate: {successes}/{len(results)} -> {folder}")
    table += [[checkpoint.name, task.env_id, trained["train/global_step"], trained["train/epoch"], arguments.execution, arguments.max_steps, *result.values()] for result in results]
    logs |= {
      **({key("sim", config.tasks, task, "success_rate"): successes / len(results)} if len(config.tasks) > 1 else {}),
      key("sim", config.tasks, task, "grid"): wandb.Video(str(folder / "grid.mp4"), caption=f"{checkpoint.name}: {successes}/{len(results)} succeeded"),
    }
  print(f"Success rate: {sum(every)}/{len(every)} over {len(runs)} task(s) -> {output}")

  with wandb.init(entity=config.wandb.entity, project=config.wandb.project, group=config.wandb.group, mode=config.wandb.mode, dir=run_dir.parent, name=run_dir.name, id=id_file.read_text().strip() if id_file.exists() else None, resume="allow") as run:
    id_file.write_text(run.id)
    run.define_metric("sim/success_rate", summary="max")  # the run's best checkpoint score, for sorting runs
    run.log({
      **{name: value for name, value in trained.items() if value is not None},  # absent when the checkpoint has no trainer_state.json
      "sim/success_rate": float(np.mean(every)),  # over every episode, whichever its task
      "sim/episodes": wandb.Table(columns=["checkpoint", "task", "step", "epoch", "execution", "max_steps", "episode", "seed", "success", "steps", "mean_reward"], data=table),
      **logs,
    })


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("run_dir", help="a training run folder, e.g. runs/push_cube")
  parser.add_argument("--checkpoint", nargs="*", default=[None], help="checkpoint-* folders, best, best_success or final, each simulated in turn; defaults to the latest checkpoint")
  parser.add_argument("--config", default="push_cube", help="the preset for runs with no wandb_id of their own")
  parser.add_argument("--episodes", type=int, default=EPISODES)
  parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
  parser.add_argument("--execution", choices=EXECUTIONS, default="ensemble", help="how a step's action comes from the predicted chunks")
  arguments = parser.parse_args()
  if len(arguments.checkpoint) == 1: main(argparse.Namespace(**{**vars(arguments), "checkpoint": arguments.checkpoint[0]}))
  else:  # a process per checkpoint: SAPIEN's GPU physics is not reliably rebuilt inside one process
    flags = ["--config", arguments.config, "--episodes", str(arguments.episodes), "--max_steps", str(arguments.max_steps), "--execution", arguments.execution]
    for checkpoint in arguments.checkpoint: subprocess.run([sys.executable, "-m", "ragmani.simulate", arguments.run_dir, *flags, "--checkpoint", checkpoint], check=True)
