# syntax=docker/dockerfile:1
# ragmani's image for Nautilus Jobs and sparky, amd64 and arm64: CUDA 13, ManiSkill's Vulkan rendering, every locked dependency in /opt/venv, GPU FAISS and the I-JEPA weights.
# Build from act/: docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t gitlab-registry.nrp-nautilus.io/ragmani/ragmani:<commit> --push .

ARG CUDA=13.0.3
ARG UV=0.13.3

FROM ghcr.io/astral-sh/uv:${UV} AS uv

# arm64 builds GPU FAISS from source, since PyPI has no aarch64 faiss-gpu
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS faiss-arm64
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git libopenblas-dev swig && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /build
COPY docker/faiss.sh docker/faiss.sh
RUN uv venv --python 3.11 .venv && uv pip install --python .venv numpy && bash docker/faiss.sh

# amd64 takes faiss-gpu from the lock, so its stage only leaves an empty wheels folder
FROM ghcr.io/astral-sh/uv:${UV}-alpine AS faiss-amd64
RUN mkdir -p /build/wheels

FROM faiss-${TARGETARCH} AS faiss

# base, not runtime: torch's wheels already ship CUDA's libraries, and the arm64 FAISS build links against those
FROM nvidia/cuda:${CUDA}-base-ubuntu24.04
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/opt/hf \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git libegl1 libgl1 libglib2.0-0 libgomp1 libopenblas0 libsm6 libvulkan1 libx11-6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --from=uv /uv /uvx /usr/local/bin/

# the locked environment, so a Job's `uv run --frozen` finds it installed and starts at once
WORKDIR /opt/act
COPY pyproject.toml uv.lock ./
COPY detr detr
# in four layers, none too large for NRP's registry to take in one upload: CUDA's libraries, cuDNN, torch, then the rest; each sync installs only what the others leave out
RUN uv export --frozen --no-hashes --no-header --no-annotate --format requirements-txt | sed -n 's/^\([A-Za-z0-9_][A-Za-z0-9_.-]*\)[ =@;].*/\1/p' > /tmp/packages
RUN --mount=type=cache,target=/root/.cache/uv UV_COMPILE_BYTECODE=0 uv sync --frozen --inexact $( (grep -vE '^nvidia-' /tmp/packages; grep -E '^nvidia-cudnn' /tmp/packages) | sed 's/^/--no-install-package /')
RUN --mount=type=cache,target=/root/.cache/uv UV_COMPILE_BYTECODE=0 uv sync --frozen --inexact $(grep -vE '^nvidia-cudnn' /tmp/packages | sed 's/^/--no-install-package /')
RUN --mount=type=cache,target=/root/.cache/uv UV_COMPILE_BYTECODE=0 uv sync --frozen --inexact $(grep -vE '^(torch|torchvision|triton)$' /tmp/packages | sed 's/^/--no-install-package /')
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen && rm /tmp/packages

# two aarch64 wheel faults: SAPIEN bundles a glibc 2.28 librt that fails on newer systems (haosulab/SAPIEN#284), so it points at the system one;
# NVIDIA's cusparselt records the tag manylinux2014_sbsa, which uv reads as another platform and reinstalls on every sync, so it records aarch64
RUN if [ "$(uname -m)" = aarch64 ]; then \
      for bundled in /opt/venv/lib/python3.11/site-packages/sapien.libs/librt-*.so; do [ -e "$bundled" ] && ln -sf /lib/aarch64-linux-gnu/librt.so.1 "$bundled"; done; \
      sed -i 's/^Tag: py3-none-manylinux2014_sbsa$/Tag: py3-none-manylinux2014_aarch64/' /opt/venv/lib/python3.11/site-packages/nvidia_cusparselt_cu13-*.dist-info/WHEEL; \
    fi

# on arm64, the GPU build replaces the lock's faiss-cpu files; mounted, so the wheel itself stays out of the image
RUN --mount=type=bind,from=faiss,source=/build/wheels,target=/wheels if ls /wheels/faiss-*.whl > /dev/null 2>&1; then uv pip install --python /opt/venv/bin/python --reinstall --no-deps --no-cache /wheels/faiss-*.whl; fi

# SAPIEN's GPU physics library for this architecture, which it otherwise downloads into ~/.sapien in every pod
# PHYSX is what SAPIEN 3.0.3's sapien.physx.version() reports; importing sapien here would crash without a GPU, and a SAPIEN wanting another version downloads its own
ARG PHYSX=105.1-physx-5.3.1.patch0
RUN python - "${PHYSX}" <<'EOF'
import io, platform, sys, urllib.request, zipfile
from pathlib import Path

name = {"x86_64": "linux-so", "aarch64": "linux-aarch64-so"}[platform.machine()]
url = f"https://github.com/sapien-sim/physx-precompiled/releases/download/{sys.argv[1]}/{name}.zip"
zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=600).read())).extractall(Path.home() / ".sapien" / "physx" / sys.argv[1])
EOF

# the sim checks and simulate encode live frames with it
ARG ENCODER=jmtzt/ijepa_vith16_1k
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${ENCODER}', allow_patterns=['*.json', '*.safetensors'])"
