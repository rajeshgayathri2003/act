# ragmani

Retrieval-augmented ACT on ManiSkill: encode demos into embeddings, train a policy that retrieves demo steps and attends over them, then score it in the sim. It restructures the `rag_model.py` pipeline; the files at the repo root are untouched, and ragmani reads their datasets.

## Setup

```sh
uv sync
uv run wandb login
```

Run every command from `act/`.

| Needs | Why |
|---|---|
| NVIDIA GPU, about 16 GB, driver 580+ | torch comes from the CUDA 13.0 index |
| NVIDIA's Vulkan driver, `NVIDIA_DRIVER_CAPABILITIES` including `graphics` | ManiSkill renders through Vulkan |
| `DATA_DIR` (default `../data`) | each task's demos (`<env>/trajectory...h5` and its `.json`) and encoded dataset |
| `WANDB_API_KEY` or `wandb login` | logging; simulate also reloads a run's config from W&B |
| `HF_HOME`, optionally | the image encoder's cache; on Kubernetes keep it, the venv and `UV_CACHE_DIR` on local disk |

## Workflow

```sh
uv run python -m ragmani.data.encode push_cube                   # demos -> encoded dataset
uv run python -m ragmani.train push_cube --run_dir runs/my_run   # any field as a flag, e.g. --train.lr 3e-4
uv run python -m ragmani.simulate runs/my_run --checkpoint best  # success rate and video
```

The first word is a preset from `config.py`'s `PRESETS`; `<preset> --help` lists every field.

A run folder holds `checkpoint-N/`, `best/` (lowest eval loss), `best_success/` (highest training-time success), `final/`, the `simulate/` videos and `wandb_id`. Simulate rebuilds the model from the config W&B stored for the run, so an offline run needs `wandb sync` first.

Judge a change by simulate, not eval loss: the arm's state alone predicts most of the action L1.

## Model

Image and state MLPs encode the frame and `QueryMLP` turns it into orthogonal queries. Each query searches the memory with `W q` and attends over the rows it fetched with `W^T W q`, so the action loss trains the search. A row's key is its image and state features, its value what its demo did next. The ACT-style head reads the attention's tokens and the raw state. Rows from the frame's own demo are never retrieved.

## Layout

| Path | What |
|---|---|
| `config.py` | settings, presets, run loader |
| `data/` | demos, encoding, datasets and their splits, retrieval |
| `models/` | image encoders and `RagAct` |
| `train.py` | training and its bookkeeping |
| `wandb_metrics.py` | the W&B run, what each metric means, and the W&B Trainer callbacks |
| `simulate.py` | closed-loop simulation |
| `launch.py`, `nautilus.sh` | queue a run as a Nautilus Job, and the Job's command |
| `visuals.py` | frames, captions and video |
