---
name: todo
description: Open work on ragmani, the rework of rag_model.py in act/ragmani
kind: todo
---

# Todo

<!-- map: generated. Edit the note, not this; after a manual edit run `agent remap` -->

| section | line |
|---|---|
| Next | 19 |
| Ideas | 24 |

<!-- /map -->


## Next

- [ ] Test the self-retrieval explanation in [[learned]]: 100 epochs with `train.split=episode train.exclude_own_episode=true`, simulate it, compare with old_state_100 (0/10).
- [ ] Docker image for ragmani (teammate); the README's Requirements table is the spec.

## Ideas

Comparisons to show a change actually improves success in simulate, not just eval loss.

- Exact retrieval against FAISS IVF on the real embeddings: speed, recall, and whether it changes success.
- Temporal ensembling of action chunks at simulate time, as ACT does, against executing only the first action.
- Separate key encoders against the shared ones (rag_model.py built unused `_value` encoders).
- Raw against normalized I-JEPA embeddings, and SigLIP against I-JEPA.
