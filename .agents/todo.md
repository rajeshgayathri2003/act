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
| Ideas | 23 |

<!-- /map -->


## Next

- [ ] Docker image for ragmani (teammate); the README's Requirements table is the spec.

## Ideas

Comparisons to show a change actually improves success in simulate, not just eval loss.

- Temporal ensembling against executing only the first action (`simulate --execution first`), once a model succeeds often enough for the difference to show.
- Generalize past PushCube to other ManiSkill tasks (pulling, opening) once one task works, with more simulate layouts then.
- tyro as the config tool: typed dataclasses per class, experiments as Python presets with `--help`. Not worth it while the yaml plus `section.key=value` overrides stay small; revisit when config mistakes cost runs or presets multiply.
- Separate key encoders against the shared ones (rag_model.py built unused `_value` encoders).
- SigLIP and other encoders against I-JEPA (a teammate is comparing encoders).
- Retrieve by motion, not one frame: key each row by an embedding of several consecutive frames (I-JEPA over a short clip, or the current and previous frame's embeddings together), so a lookup finds demos that were moving the same way along the same path. One frame cannot show direction, and state plus step is not the same thing. Needs re-encoding the dataset with frame windows.
