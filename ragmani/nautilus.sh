#!/usr/bin/env bash
# The command of the W&B Launch job ragmani-train: inside a Nautilus Job, installs the locked environment and trains one run.
# Launch mounts the repo at the job's commit and calls `bash ragmani/nautilus.sh <config> [section.key=value ...]`; ragmani/launch.py names the run and passes both.
set -euo pipefail
config="${1:?the config comes first; ragmani/launch.py passes it}"
shift

# a folder another run already holds gets this run's id appended; this run's own folder means a restart, which resumes
run_dir="/marco-west/runs/${WANDB_NAME:?queue runs with ragmani/launch.py, which names them}"
run_id="${WANDB_RUN_ID:?Launch sets it}"
if [[ -e "$run_dir" && "$(cat "$run_dir/wandb_id" 2> /dev/null)" != "$run_id" ]]; then run_dir="$run_dir-$run_id"; fi

# venv and caches on the pod's disk (/work): many small files, which the network volume handles worse
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH" UV_PROJECT_ENVIRONMENT=/work/venv UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/work/python
uv sync --frozen

# the configs' task paths, ../data and ../robot-manipulation, resolve to the volume, which mirrors that layout
for folder in data robot-manipulation; do [[ -e "../$folder" ]] || ln -s "/marco-west/$folder" "../$folder"; done

# runs on the volume, so checkpoints outlive the pod; overrides after these win
exec uv run --frozen python -m ragmani.train.rag_act "$config" \
  paths.cache=/work/hf \
  "paths.run_dir=$run_dir" \
  "$@"
