---
name: context
description: What the act repo and its ragmani package assume, for changing them correctly
kind: context
---

# Context

A lab-shared fork of tonyzhaozh/act. `ragmani/` restructures the RAG pipeline around `rag_model.py`; the root files (`rag_model.py`, `rag_retrieval/`, `ijepa_encoder.py`, `test.py`, the ALOHA sim) stay as they are, but `pyproject.toml` installs only what ragmani uses, so their `detr` is not a dependency.

- Run from `act/` as modules: `uv run python -m ragmani.<module>`. Imports are absolute.
- Settings are typed dataclasses in `ragmani/config.py`, one section per class; experiments are `PRESETS`, overridden by tyro flags after the preset (`push_cube --train.lr 3e-4`, `--tasks.1.dataset ...`). Choices are name tables (`ENCODERS`, `SPLITS`, `EXECUTIONS`) or enums, not `if` chains.
- Task paths sit under `DATA_DIR` (default `../data`); the model cache is `HF_HOME`. In a Job, `ragmani/queue.yaml` sets both, plus `RUNS_DIR`, on the queue's volume; the code holds no Nautilus paths.
- `python -m ragmani.launch` does all of Nautilus: `setup` builds your queue `ragmani-$RAGMANI_USER` from `queue.yaml` and `.env`, `agent` runs its agent, `<run> --preset <name> [flags]` queues a run, and `job` starts it inside the pod (README, Launching on Nautilus).
- `tasks` trains one model on every task's rows in one memory; `data.combine` adds `task` and `demo` columns; simulate shares a fixed episode count across tasks.
- Models only compute; losses and logging live in `train.py`.
- Runs open W&B through `wandb_metrics.start`; a new metric gets one line in `wandb_metrics.METRICS` (per-task ones in `TASK_METRICS`, named through `wandb_metrics.key`).
- A run's config lives in W&B; `simulate` reloads it via the run folder's `wandb_id`. Offline runs cannot be reloaded until synced.
- Before a long run, or after changing training code, smoke it locally with `--train.epochs 1 --train.simulate_epochs 1 --train.simulate_episodes 2`.
- Judge a change by simulate, not eval loss.
- Checkpoints from `rag_model.py` or from before the `simplify` rework do not load into `RagAct`, and datasets from its encoders (`ijepa_encoder.py`, `data_encoder*.py`) do not load either; re-encode them with `ragmani.data.encode`.
- Dataset and live frames must use the same encoder settings, `encoder.precision` included; changing one means re-encoding.
- Simulate's reported scores use seed 42's layouts, training checks `CHECK_SEED`'s, so `best_success` never sees the reported layouts.
- Python: two-space indent, no `Author:` line ([[python]]).
- `.agents/.local/` is per-machine and gitignored.
