# Vocab pruning (structured, embeddings) — the size lever

**Method:** scan the corpus, keep only tokens that appear, prune the embedding + LM/score
rows, ship a `remap.npy` applied to input_ids at inference. Part of the submission recipe
(`prune_vocab` + remap + parity gate). Index: [results.md](results.md).

| Model | Status |
|---|---|
| granite | ✅ **IN USE** — biggest single size lever (embeddings = 64.6% of params, 201M of 312M) |
| bge-m3 / qwen3 | ✅ proven (submission recipe) |

## Why it's the granite size story
granite's mass is the 262k×768 embedding table (64.6%), which is ~0 FLOPs (a lookup). So
vocab-prune is almost pure **size** reduction, irrelevant to speed. Everything speed-related
lives in the 22-layer stack (see [depth-pruning.md](depth-pruning.md),
[width-pruning.md](width-pruning.md)).

## Notes
- Pruned tokenizer + `remap.npy` are model-independent → **reuse** across same-backbone +
  same-variant submissions ([[reuse-pruned-tokenizer]]); don't re-run the 70k-token rescan.
- ⚠️ Record retained-vocab fraction + exact MB saved for granite at the next build (open).
- The shipped t031 zip is **full-vocab fp16** (629MB) — NOT yet vocab-pruned, so there's an
  easy size cut available there if needed.
