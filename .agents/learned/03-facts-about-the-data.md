---
name: facts-about-the-data
description: Facts about the data
kind: learned
---

# Facts about the data

- **Each action appears in up to 20 rows' chunks**, and the first 19 steps of a demo in fewer, so statistics over all chunk steps under-weight demo starts: the action mean's x was 0.0229 that way against 0.0184 over each step once (68,978 steps). `data.statistics` uses each row's first action.
- **I-JEPA embeddings of different frames are 0.994 similar, and true neighbours differ by about 0.0003**, so tiny numeric changes reorder them: bf16 embeddings of live frames (cosine 0.999935 to fp32) found the same nearest row as fp32 only 78% of the time against the fp32 dataset (300 frames). The dataset and live frames must share `encoder.precision`.
- **I-JEPA ViT-H dominates simulate time**: on sparky, fp32 takes about 112 ms per frame at any batch size (the GPU is compute-bound), bf16 about 27 ms. FAISS search (2-5 ms), the policy (6-9 ms) and render plus physics (about 4 ms per env) barely count.
- **FAISS IVF-SQ8 caps recall@6 at about 0.83 on the real embeddings** however many cells it probes (nprobe 8 to 64); IVF-flat reaches 0.985. The neighbours' action quality is the same for all (L1 0.290), so the misses are near-ties.
- **PushCube demo `env_states` hold T+1 states for T actions**; the last is the true final state, so no step is needed for the final `next_state`.
