---
name: python
description: Python layout in the shared lab repo
applies: *.py, *.pyi, *.ipynb, pyproject.toml
---

# Python in this repo

Module docstrings carry no `Author:` line. This is a shared lab repo; authorship lives in git history.

## Names in model math

Inside a model's forward pass, tensors take the short names of PyTorch and the papers rather than full words: `q`, `k`, `v` for attention queries, keys and values; `h_obs` for an encoded observation; `z` for latent tokens; `q_fetch`, `q_stack`, `sas` as rag_model.py names them; `idx` and `topk_idx` for indices; `attn_out`, `attn_bias`. A trailing comment gives the shape where it is not obvious. Everything outside that math (modules, functions, config keys, loops and the rest of the code) keeps full words.

Layer names (`image_encoder`, `query_fetch`, …) are checkpoint keys: rename one only with a mapping that keeps existing checkpoints loading.
