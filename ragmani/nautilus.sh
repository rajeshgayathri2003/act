#!/usr/bin/env bash
# The command of the W&B Launch job ragmani-train: inside a Nautilus Job, installs the locked environment and trains one run.
# Launch mounts the repo at the job's commit and calls `bash ragmani/nautilus.sh [section.key=value ...]`; ragmani/launch.py names the run and passes the overrides.
set -euo pipefail

# a folder another run already holds gets this run's id appended; this run's own folder means a restart, which resumes
run_dir="/marco-west/runs/${WANDB_NAME:?queue runs with ragmani/launch.py, which names them}"
if [[ -e "$run_dir" && "$(cat "$run_dir/wandb_id" 2> /dev/null)" != "$WANDB_RUN_ID" ]]; then run_dir="$run_dir-$WANDB_RUN_ID"; fi

# venv and caches on the pod's disk (/work): many small files, which the network volume handles worse
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH" UV_PROJECT_ENVIRONMENT=/work/venv UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/work/python
uv sync --frozen

# data and runs on the volume, so checkpoints outlive the pod; overrides after these win
exec uv run --frozen python -m ragmani.train.rag_act push_cube \
  paths.dataset=/marco-west/data/push_cube_jepa \
  paths.traj=/marco-west/robot-manipulation/PushCube-v1/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  paths.cache=/work/hf \
  "paths.run_dir=$run_dir" \
  "$@"
