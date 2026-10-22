---
name: todo.archive
description: Finished ragmani work, kept for why the code looks the way it does
kind: todo.archive
---

# Todo archive

- [x] tyro as the config tool: typed dataclasses per section and experiments as Python presets with `--help`, built in the `simplify` rework.
- [x] Run the nine-arm A/B sweep on Nautilus (W&B sweep `sy54wr1m`, 2026-10-12, 5 h 30 min on one A10): searching by the frame's embedding was the clear win over the learned query; most arms peaked at epoch 30-40 then declined ([[learned]]). The sweep file and its `test=` handling were removed afterwards; a fixed list of runs needs no sweep.
- [x] Test the self-retrieval explanation: `episode_split_100` (episode split, own demo excluded) still got 0/10 on 2026-10-11, so self-retrieval was not the cause; retrieval collapse was ([[learned]]).
- [x] Exact retrieval against FAISS IVF on the real embeddings: IVF-SQ8 recall@6 0.83, IVF-flat 0.985, same neighbour action quality; FAISS stays, on the GPU on x86 (`faiss-gpu`).
- [x] Temporal ensembling at simulate time, as ACT does: the default since 2026-10-11 (`--execution first` keeps the old behaviour).
- [x] Raw against centred (and normalized) I-JEPA embeddings for retrieval: centring changed nothing (action L1 0.288 against 0.294).
- [x] Commit ragmani and push `marco-dev`, so teammates can clone it. Committed as f4239bf and pushed to rajeshgayathri2003/act on 2026-10-10; that push also settled that Marco can push there.
