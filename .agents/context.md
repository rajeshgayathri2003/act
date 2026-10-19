---
name: context
description: What the act repo and its ragmani package assume, for changing them correctly
kind: context
---

# Context

A lab-shared fork of tonyzhaozh/act. `ragmani/` restructures the RAG pipeline around `rag_model.py`; the root files (`rag_model.py`, `rag_retrieval/`, `ijepa_encoder.py`, `test.py`, the ALOHA sim) stay as they are.

- Run from `act/` as modules: `uv run python -m ragmani.<module>`. Imports are absolute.
- Settings are typed dataclasses in `ragmani/config.py`, one section per class; experiments are `PRESETS`, overridden by tyro flags after the preset (`push_cube --train.lr 3e-4`, `--tasks.1.dataset ...`). Choices are name tables (`ENCODERS`, `SPLITS`, `EXECUTIONS`) or enums, not `if` chains.
- Task paths sit under `DATA_DIR` (default `../data`); the model cache is `HF_HOME`. `nautilus.sh` sets both for Jobs; `python -m ragmani.launch <run> --preset <name> [flags]` queues one.
- `tasks` trains one model on every task's rows, episodes offset by `data.TASK_STRIDE`; simulate shares a fixed episode count across tasks.
- Models only compute; losses and logging live in `train.py`.
- A run's config lives in W&B; `simulate` reloads it via the run folder's `wandb_id`. Offline runs cannot be reloaded until synced.
- Before a long run, or after changing training code, smoke it locally with `--train.epochs 1 --train.simulate_epochs 1 --train.simulate_episodes 2`.
- Judge a change by simulate, not eval loss.
- Checkpoints from `rag_model.py` or from before the `simplify` rework do not load into `RagAct`; its datasets still load through `data.load`.
- Dataset and live frames must use the same encoder settings, `encoder.precision` included; changing one means re-encoding.
- Simulate's reported scores use seed 42's layouts, training checks `CHECK_SEED`'s, so `best_success` never sees the reported layouts.
- Python: two-space indent, no `Author:` line ([[python]]).
- `.agents/.local/` is per-machine and gitignored.
