---
name: context
description: What the act repo and its ragmani package assume, for changing them correctly
kind: context
---

# Context

The repo is a fork of tonyzhaozh/act shared by the lab. `ragmani/` restructures the RAG pipeline that grew around `rag_model.py`; the files at the root (`rag_model.py`, `rag_retrieval/`, `ijepa_encoder.py`, `test.py`, the ALOHA sim) stay as they are, since others work from them.

- Run from `act/`, as modules: `uv run python -m ragmani.<module> ...`. Imports are absolute from the package root.
- One experiment is one yaml in `ragmani/config/`; any value overrides as `section.key=value`. A class takes its own section, and choices between implementations are name tables (`ENCODERS`, `SPLITS`, `EXECUTIONS`), not `if` chains.
- Models only compute. Losses, the truth schedule, splits and logging live in `train/`; encoders and retrieval are passed in.
- Training saves the run's full config to `runs/<name>/config.yaml`; `simulate` rebuilds the model from it, so a run stays reproducible however the yaml changes.
- "Eval" is the Trainer's held-out loss; "simulate" is the closed-loop success rate in the sim. Judge a change by simulate.
- `push_cube.yaml` follows `rag_model.py` at 8b9a9e3 except where the README lists a difference; each switch's comment names the alternative. Datasets from `rag_model.py` must keep loading through `data.load`; its checkpoints do not load into `RagAct`.
- The dataset and live frames go through the same encoder with the same settings, `encoder.precision` included; changing one means re-encoding.
- `simulate` loads a run's saved `config.yaml` with `push_cube.yaml` as defaults, and `config.RENAMED` maps keys older runs saved under other names; a renamed key gets an entry there.
- `simulate.episodes` runs every layout at once in ManiSkill's parallel envs (GPU physics); the run scores use seed 42's layouts, training-time checks `CHECK_SEED`'s, so choosing `best_success` never sees the reported layouts.
- Python here: two-space indent, no `Author:` line ([[python]] in `standards/`).
- `.agents/.local/` holds notes for one machine and is gitignored.
