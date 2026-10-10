# ragmani

Retrieval-augmented ACT on ManiSkill: encode demos into embeddings, train a policy that retrieves similar demo steps and attends over them, then run it in the sim and record success and video. It restructures the `rag_model.py` pipeline and what was tried around it into one package that is easier to read and for several people to work on. The original files at the repo root are untouched, and ragmani reads their datasets and checkpoints.

## Setup

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/), Astral's Python package manager; `curl -LsSf https://astral.sh/uv/install.sh | sh` installs it.

```sh
uv sync
uv run wandb login
```

Run every command from `act/`, the repo root.

## Requirements

What a machine or container needs, however it is set up:

| Need | Why |
|---|---|
| Linux x86_64 with an NVIDIA GPU of about 16 GB or more, driver 580 or newer | torch comes from the CUDA 13.0 index (`pyproject.toml`); an A10, L4 or RTX 3090 is plenty |
| NVIDIA's Vulkan driver in the container (`/usr/share/vulkan/icd.d/nvidia_icd.json`) and `NVIDIA_DRIVER_CAPABILITIES` including `graphics` | ManiSkill renders through Vulkan; without it, encode and simulate fail at `gym.make` |
| About 8 CPUs | dataloader workers, and the sim's physics runs on the CPU |
| The demos at `paths.traj` | `PushCube-v1/trajectory.state.pd_ee_delta_pos.cpu.h5` and the `.json` beside it |
| The encoded dataset at `paths.dataset` | made by `ragmani.data.encode`, about 2.5 h for PushCube on one GPU |
| `WANDB_API_KEY` in the environment, or `wandb login` | training and simulate log to W&B; `wandb.mode=disabled` turns it off |

On Kubernetes with a network volume such as Nautilus's: keep the venv, uv's cache (`UV_CACHE_DIR`) and the model cache (`paths.cache`) on the pod's local disk. The volume writes at about 0.2 MB/s, so installs and the 2.5 GB I-JEPA download there take hours; code, data and `runs/` are fine on it.

## Workflow

```sh
uv run python -m ragmani.data.encode push_cube                      # demos -> encoded dataset, once per dataset
uv run python -m ragmani.train.rag_act push_cube paths.run_dir=runs/my_run
uv run python -m ragmani.simulate runs/my_run --checkpoint best     # success rate and video in the sim
```

Any config value can be overridden as `section.key=value`, e.g. `train.epochs=100 train.lr=1e-4`.

**Encode** replays each demo, renders every step, embeds the frames and writes one row per step to `paths.dataset`.

**Train** writes everything for a run to `runs/<name>/` and logs to W&B:

| File | What |
|---|---|
| `config.yaml` | the full config the run used |
| `checkpoint-N/` | every `train.save_epochs` epochs |
| `best/` | the model at its lowest eval loss |
| `final/` | the model at the end |
| `simulate/training/` | one full sim episode every `train.simulate_epochs` epochs |

**Simulate** runs 10 episodes with a checkpoint (`best`, `final`, `checkpoint-N`; the latest by default) and logs the success rate, an episode table and videos to the same W&B run.

Two evaluations exist. **Eval** is the Trainer's held-out loss (`eval/loss`): predicted actions against demo actions. **Simulate** measures whether the policy actually pushes the cube to the goal (`simulate/...`). Loss can fall while success stays at zero, so simulate is the number that decides; with `train.split=frame` and the row's own demo retrievable, eval loss has fallen sharply while success went to zero.

## Config

`config/push_cube.yaml` holds one experiment. Its values reproduce `rag_model.py` at commit 8b9a9e3, and each switch's comment names the alternative (`train.split`, `train.exclude_own_episode`, `train.truth_coin`, `model.draw`, `train.lr_scheduler`). The one difference it cannot reach is embedding normalization: `rag_model.py`'s data was not normalized, so matching it needs `encoder.normalize=false` when encoding.

Datasets and checkpoints from `rag_model.py` load directly; `data.load` and `RagAct.load` convert them.

## Layout

| Path | What |
|---|---|
| `config/` | experiment yamls and their loader |
| `data/` | demo reading, encoding, dataset loading, retrieval, train/eval splits |
| `models/` | image encoders and the `RagAct` policy |
| `train/rag_act.py` | training |
| `simulate.py` | closed-loop simulation |
| `visuals/` | frames and video |
