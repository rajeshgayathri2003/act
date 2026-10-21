#!/usr/bin/env bash
# Builds FAISS with GPU support into a wheel under wheels/, for this machine's architecture: aarch64 (sparky, which PyPI's faiss-gpu skips) or x86_64.
# Run from act/ as `bash docker/faiss.sh [--dry-run]`; the image's Dockerfile runs it too. Needs the CUDA toolkit, cmake, swig and OpenBLAS (libopenblas-dev).
set -euo pipefail

version="${FAISS_VERSION:-v1.15.1}"
cuda="${CUDA_HOME:-/usr/local/cuda}"
case "$(uname -m)" in
  aarch64) optimized=sve; default_architectures="90;121" ;;    # GH200 and the GB10
  x86_64) optimized=avx2; default_architectures="75;80;86;89" ;;  # the Turing-to-Ada cards queue.yaml allows
  *) echo "no FAISS build set up for $(uname -m)" >&2; exit 1 ;;
esac
architectures="${CUDA_ARCHITECTURES:-$default_architectures}"
python="${PYTHON:-$PWD/.venv/bin/python}"   # the Python the wheel is built for, with numpy installed
include="$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"  # asked of the interpreter, since CMake misses uv's Python inside the image
numpy_include="$("$python" -c 'import numpy; print(numpy.get_include())')"
work="${WORK_DIR:-build/faiss}"
dry_run=""
[[ "${1:-}" == "--dry-run" ]] && dry_run=1

run() {
  if [[ -n "$dry_run" ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

[[ -d "$work/src" ]] || run git clone --depth 1 --branch "$version" https://github.com/facebookresearch/faiss.git "$work/src"

# a second library optimized for this CPU family sits beside the generic one; faiss loads it when the CPU supports it
run cmake -S "$work/src" -B "$work/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_C_API=OFF \
  -DFAISS_OPT_LEVEL="$optimized" \
  -DBUILD_TESTING=OFF \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="$architectures" \
  -DPython_EXECUTABLE="$python" \
  -DPython_INCLUDE_DIR="$include" \
  -DPython_NumPy_INCLUDE_DIR="$numpy_include"
run cmake --build "$work/build" --parallel "$(nproc)" --target swigfaiss "swigfaiss_$optimized"

# the package setup.py writes is tagged for any platform; tag it for this one, so no other machine installs it
run uv build --wheel "$work/build/faiss/python" --out-dir wheels
tag="$("$python" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"  # the interpreter the module was built for
for wheel in wheels/faiss-*-py3-none-any.whl; do
  [[ -e "$wheel" || -n "$dry_run" ]] || { echo "no wheels/faiss-*-py3-none-any.whl was built" >&2; exit 1; }
  run uvx --from wheel wheel tags --remove --python-tag "$tag" --abi-tag "$tag" --platform-tag "linux_$(uname -m)" "$wheel"
done
run ls wheels
