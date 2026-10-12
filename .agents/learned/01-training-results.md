---
name: training-results
description: Training results
kind: learned
---

# Training results

- **Many PushCube layouts are easy**: a model trained 2 epochs (`query=observation`, normalized) succeeded on 8/20 layouts (seed 42, GPU physics, 128-step cap, 2026-10-12). Read any success rate against a barely trained model on the same layouts, not against 0.
- **`observation_100` (search by the frame's own embedding, normalized, lr 1e-4, no clipping, weight decay 1e-4, 100 epochs) succeeded on 6/10 (best, epoch 46) and 5/10 (final)**, seed-42 layouts, CPU physics, 400-step cap; `episode_split_100` under rag_model.py's settings got 0/10. Five settings changed at once, so which one mattered is untested. Eval loss bottomed at 0.261 (epoch 46) and rose to 0.304 while success held, so eval loss does not pick the better checkpoint.
- **Retrieval with the learned query collapses to the same 20-40 rows for every input, which is what broke rag_model.py's policy**: the search returns indices, so `query_fetch` never gets a gradient and keeps its random weights (a gradient watch reached every other parameter). Its retrieved frames are less similar to the query (cosine 0.988) than a random row (0.994). Self-retrieval was not the cause: with the own demo excluded the policy still got 0/10. Searching by the frame's own embedding keeps 0.32 of each batch's retrievals distinct, at similarity 1.000.
- **Adding the search score to attention (`search_scores_in_attention`) trains `query_fetch` but only reaches random-level retrieval** in 10 epochs: similarity 0.990 to 0.995, distinct 0.004 to 0.008.
- **Nearest neighbours by the raw I-JEPA embedding have useful actions**: L1 to the query's next 5 standardized actions 0.294 against 0.759 for random rows (2000 queries, own demo excluded). Centring the embeddings changes nothing (0.288); searching by state alone is worse (0.554) and collapses to 783 rows, and every weight of state added to the image key makes it worse.
- **The padding BCE is about a quarter of the eval loss** after 2 epochs (0.142 of 0.559), so the total can fall through the padding prediction alone; `eval/action_l1` and `eval/pad_bce` are logged apart.
