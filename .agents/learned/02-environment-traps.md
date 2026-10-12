---
name: environment-traps
description: Environment traps
kind: learned
---

# Environment traps

- **transformers 5 removed `TrainingArguments(warmup_ratio=)`**; `warmup_steps` takes a float below 1 as a share of the steps.
- **`wandb[launch]` downgrades protobuf to 6.x for the whole lock**, through its Google packages. A Kubernetes queue's agent still imports Azure and requires `awscli` for any kubeconfig, so the `launch` group lists those five; check with `import wandb.sdk.launch.runner.kubernetes_runner`.
- **cogrob members cannot install the W&B Launch Helm chart** (no Roles or RoleBindings), so the agent runs off-cluster on the local kubeconfig.
- **`kubectl exec` streams reset on large copies** (369 MB failed at about 2 MB). Copy data as a tar split into 50 MB pieces with `kubectl cp`, check sizes, rejoin on the pod and verify a sha256. Keep the venv, uv cache and HF cache on the pod's local disk: many small files, which a network filesystem handles worse (not measured).
- **Plain `uv sync` works on Nautilus x86** (drivers 595/615, CUDA 13.2/13.4, cu130 torch): under a minute with a local cache. The image has NVIDIA's Vulkan ICD, so SAPIEN renders on the GPU.
- **faiss-cpu and torch abort in one process on macOS** (`OMP: Error #15`): both pip wheels bundle LLVM `libomp`. `KMP_DUPLICATE_LIB_OK=TRUE` alone segfaults in the IVF build; with `OMP_NUM_THREADS=1` as well it runs and gives correct results (FAISS flat equals numpy brute force on 500 queries). On Linux (x86 and aarch64) the same pair ran in one process without the error.
- **`datasets` 5.x `dataset["column"]` returns a lazy `Column`**, not an array; use `dataset.with_format("numpy")[:]` for a dict of arrays.
- **`pd_ee_delta_pos` fails at `gym.make` on macOS without the `pin` package**: the EE controller builds a Pinocchio IK model and SAPIEN's macOS wheel ships without it (`'NoneType' object is not callable` in `kinematics.py`). `pin>=4.1.0` is in the toml and fixes it; replay in `ragmani/data/encode.py` uses `pd_joint_pos` regardless, since it never steps.
- **pyarrow 26 refuses NumPy 1.x at import, and no metadata declares it**, so uv resolves pyarrow 26 next to a `numpy<2` pin and imports fail. ragmani runs on NumPy 2; do not pin `numpy<2` again.
- **`transformers` 5.x `get_image_features` returns an output object, not a tensor**; the SigLIP embedding is `.pooler_output`.
- **gymnasium 1.x wrappers do not forward attributes**, so `set_state_dict` needs `env.unwrapped`.
