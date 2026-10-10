---
name: learned
description: What ragmani's runs showed so far, environment traps on Nautilus, macOS and Linux, and checks of the data against rag_model.py's
kind: learned
---

# Learned

<!-- map: generated. Edit the note, not this; after a manual edit run `agent remap` -->

| section | line |
|---|---|
| Training results (open) | 20 |
| Environment traps | 24 |
| Facts about the data | 39 |

<!-- /map -->


## Training results (open)

- **Under rag_model.py's settings, eval loss and sim success diverge**: old_state_100 (frame split, own episode retrievable, truth faded over 100 epochs) took eval loss 0.25 to 0.12 after epoch 70 while simulate fell from 1/10 (10-epoch smoke) to 0/10 and per-episode mean reward from 0.11-0.19 to 0.006-0.03 in eight episodes (0.11 in the other two). Suspected: the model learns to copy retrieved near-duplicate frames of its own demo once the truth fades. Untested; the check is the same run with `train.split=episode train.exclude_own_episode=true`.

## Environment traps

- **The Nautilus PVC (CephFS) wrote at about 0.1-0.2 MB/s** when unpacking a 369 MB dataset (about 180 MB in 25 min, measured once); a uv cache of about 7 GB there would take hours by the same rate (not measured, the sync was stopped). Keep the venv (`UV_PROJECT_ENVIRONMENT`), `UV_CACHE_DIR` and the HF cache (`paths.cache`) on the pod's local disk; code, data and runs can live on the volume.
- **`kubectl exec` streams reset on large copies** (369 MB failed at about 2 MB). Send 50 MB pieces with `kubectl cp`, check sizes, rejoin and verify a sha256.
- **Plain `uv sync` works on Nautilus x86** (drivers 595/615, CUDA 13.2/13.4, cu130 torch): under a minute with a local cache. The image has NVIDIA's Vulkan ICD, so SAPIEN renders on the GPU.
- **`override-dependencies` in `~/.config/uv/uv.toml` had no effect on `uv sync`** (uv 0.12, two attempts, with and without a list in the project); not checked against uv's documentation, so treat a machine-only override in user config as unreliable rather than impossible.
- **faiss-cpu and torch abort in one process on macOS** (`OMP: Error #15`): both pip wheels bundle LLVM `libomp`. `KMP_DUPLICATE_LIB_OK=TRUE` alone segfaults in the IVF build; with `OMP_NUM_THREADS=1` as well it runs and gives correct results (FAISS flat equals numpy brute force on 500 queries). On Linux (x86 and aarch64) the same pair ran in one process without the error.
- **The old encoder's relative h5 path failed to open inside its dataset generator** (`FileNotFoundError` after the env was built); the cause was not pinned down, and making the path absolute fixed it. `ragmani` config paths are already absolute.
- **`datasets` 5.x `dataset["column"]` returns a lazy `Column`**, not an array; use `dataset.with_format("numpy")[:]` for a dict of arrays.
- **`pd_ee_delta_pos` fails at `gym.make` on macOS without the `pin` package**: the EE controller builds a Pinocchio IK model and SAPIEN's macOS wheel ships without it (`'NoneType' object is not callable` in `kinematics.py`). `pin>=4.1.0` is in the toml and fixes it; replay in `ragmani/data/encode.py` uses `pd_joint_pos` regardless, since it never steps.
- **ragmani runs on NumPy 2**: with `numpy<2` dropped, NumPy 2.4.6, pyarrow 26, sapien 3.0.3, faiss 1.15.1, h5py, torch and mujoco 2.3.7 all import, and encode, train and simulate run. The ALOHA sim (mujoco, dm_control) was only imported, not run.
- **pyarrow 26.0.0 refuses NumPy 1.x at import; 25.0.0 does not.** No metadata declares it, so uv resolves pyarrow 26 next to `numpy<2`.
- **`transformers` 5.x `get_image_features` returns an output object, not a tensor**; the SigLIP embedding is `.pooler_output`.
- **gymnasium 1.x wrappers do not forward attributes**, so `set_state_dict` needs `env.unwrapped`.

## Facts about the data

- **`ragmani/data/encode.py` reproduces the old I-JEPA encoder exactly** (PushCube episodes 0-1, 143 rows): action and state diff 0, next_state diff 0 except the two final steps (0.0145, the old `env.step(actions[0])` bug), image_emb cosine 1.000000; old norms were about 23, new are 1.
- **Retrieval speed on the M3 Max, 70k random rows, 64x3 queries, k=6, own episode excluded**: exact torch on MPS 11.1 ms per batch; FAISS IVF-SQ8 on CPU, 1 thread, 75.6 ms with recall@6 0.25. Random vectors are IVF's worst case, and exclusion makes IVF fetch k + longest episode; measure real embeddings on Linux before choosing.
- **Resetting once per episode is enough for replay**: frames and qpos were identical to resetting before every step (PushCube episode 0, steps 0, 10, T-1, T; max pixel and qpos difference 0).
- **Sim qpos after `set_state_dict` equals the recorded `env_states/articulations/panda[13:22]`** exactly, at the same steps.
- **PushCube demo `env_states` hold T+1 states for T actions**; the last is the true final state, so no step is needed for the final `next_state`.
