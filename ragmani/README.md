# ragmani

Retrieval-augmented ACT on ManiSkill: encode demos into embeddings, train a policy that retrieves demo steps and attends over them, then score it in the sim. It restructures the `rag_model.py` pipeline. The files at the repo root are untouched but not installed: `pyproject.toml` lists only what ragmani uses, so their `detr` package needs `uv pip install -e detr`, and datasets from their encoders need re-encoding with `ragmani.data.encode`.

[Setup](#setup) · [Workflow](#workflow) · [Launching on Nautilus](#launching-on-nautilus) · [Model](#model) · [Layout](#layout)

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
| `HF_HOME`, optionally | the image encoder's cache |

On aarch64 Linux (sparky), PyPI has no GPU FAISS: build it once with `bash docker/faiss.sh` (needs `libopenblas-dev`) and `uv pip install --reinstall --no-deps wheels/faiss-*.whl`. Use `uv sync --inexact` there afterwards, since a plain `uv sync` removes it. NVIDIA's aarch64 cusparselt wheel records a wrong platform tag, so uv reinstalls it on every sync until it is corrected: `sed -i 's/manylinux2014_sbsa$/manylinux2014_aarch64/' .venv/lib/python3.11/site-packages/nvidia_cusparselt_cu13-*.dist-info/WHEEL`.

## Workflow

```sh
uv run python -m ragmani.data.encode push_cube                   # demos -> encoded dataset
uv run python -m ragmani.train push_cube --run_dir runs/my_run   # any field as a flag, e.g. --train.lr 3e-4
uv run python -m ragmani.simulate runs/my_run --checkpoint best  # success rate and video
```

The first word is a preset from `config.py`'s `PRESETS`; `<preset> --help` lists every field.

A run folder holds `checkpoint-N/`, `best/` (lowest eval loss), `best_success/` (highest training-time success), `final/`, the `simulate/` videos and `wandb_id`. Simulate rebuilds the model from the config W&B stored for the run, so an offline run needs `wandb sync` first.

Judge a change by simulate, not eval loss: the arm's state alone predicts most of the action L1.

## Launching on Nautilus

Long runs go to Nautilus as Jobs, all through `python -m ragmani.launch`. You queue a run from your machine, and your launch agent, also on your machine, turns it into a Job with your Kubernetes login. Each person has their own queue, `ragmani-<user>`, built from `queue.yaml` and these lines in `act/.env`:

| `.env` | Example | What it sets |
|---|---|---|
| `WANDB_API_KEY` | | your W&B key, for launching and the agent |
| `RAGMANI_USER` | `marco` | your queue `ragmani-marco`, Job names `marco-ragmani-<run>`, the `user` label |
| `RAGMANI_VOLUME` | `marco-west-vol` | your PVC in `cogrob`, mounted at `/volume`: data from `/volume/data`, runs to `/volume/runs` |
| `RAGMANI_WANDB_SECRET` | `marco-wandb` | the Kubernetes secret holding your W&B key, so restarted pods still log |

`export UV_ENV_FILE=.env` once in your shell, and every `uv run` below reads it.

Once, with `kubectl` logged in to NRP and the datasets copied to `<volume>/data` (same layout as `DATA_DIR`):

```sh
uv sync --group launch                                  # the agent's Kubernetes runner
kubectl create secret generic marco-wandb -n cogrob --from-literal=api-key="$WANDB_API_KEY"
uv run python -m ragmani.launch setup                   # creates your queue; rerun after editing queue.yaml
```

Then keep your agent running (in tmux), and queue runs:

```sh
uv run python -m ragmani.launch agent -j 5 -v
uv run python -m ragmani.launch my-run --preset push_cube --commit simplify --smoke --train.epochs 3000 --wandb.group my_tests
```

- `--smoke` first trains the same flags here for one epoch, from your checkout, and queues only if that works; check out the commit first.
- `--commit` takes a pushed branch or hash: the pod clones it from GitHub. Flags after the known ones override the preset, and launch checks them before queuing.
- The run's name is its W&B run id, Job name (`<user>-ragmani-<run>`) and folder under `/volume/runs`: lowercase, digits and dashes. Reusing a name continues that run.
- `queue.yaml` is the one Nautilus config: GPU types, excluded nodes, the 72 h cap, resources, paths. Edit it, then rerun `setup`. A commit from before `queue.yaml` existed does not run on these queues.
- Follow a run in W&B and with `kubectl get jobs,pods -n cogrob | grep <user>`. Stopping it from the W&B UI ends training cleanly.
- Jobs run `docker/Dockerfile`'s image, with the environment, GPU FAISS, SAPIEN's PhysX library and the I-JEPA weights preinstalled, for amd64 and arm64, from the public `gitlab-registry.nrp-nautilus.io/ragmani/ragmani`. After a dependency change, rebuild it with the command at the Dockerfile's top (arm64 builds natively on sparky, amd64 under emulation after `docker run --privileged --rm tonistiigi/binfmt --install amd64`), then pin the new digest in `launch.py`'s `IMAGE`.

## Model

Image and state MLPs encode the frame and `QueryMLP` turns it into orthogonal queries. Each query searches the memory with `W q` and attends over the rows it fetched with `W^T W q`, so the action loss trains the search. A row's key is its image and state features, its value what its demo did next. The ACT-style head reads the attention's tokens and combined, the image and state features, detached so the encoders learn only through retrieval. Rows from the frame's own demo are never retrieved. With several tasks, every task's rows share one memory.

## Layout

| Path | What |
|---|---|
| `config.py` | settings, presets, run loader |
| `data/` | demos, encoding, datasets and their splits, retrieval |
| `models/` | image encoders and `RagAct` |
| `train.py` | training and its bookkeeping |
| `wandb_metrics.py` | the W&B run, what each metric means, and the W&B Trainer callbacks |
| `simulate.py` | closed-loop simulation |
| `launch.py`, `queue.yaml` | Nautilus: set up your queue, run its agent, queue runs and start them in their Job; the Job template |
| `visuals.py` | frames, captions and video |
