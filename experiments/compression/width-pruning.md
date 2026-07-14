# Width pruning (structured) — remove attention heads / FFN neurons + recovery FT

**Method:** rank units by importance (FLAP fluctuation, Wanda-sp = |w|·‖x‖ activation
magnitude) → drop low-importance attention heads and/or FFN channels → recovery FT. Output =
smaller *dense* model, real GEMM shrink, speedup unconditional on any HW. Index:
[results.md](results.md).

| Model | Status |
|---|---|
| granite | 🔲 unexplored — candidate after the depth verdict |
| qwen3 | 🔲 reserved (E45 B4, lower priority than depth×FFN×attn) |

## Plan (granite, box-safe first)
1. **Importance scoring (training-free, local):** collect per-channel FFN activation
   magnitude + per-head attention output norm over a val slice → rank. Mirrors the
   BI/zero-shot approach in [depth-pruning.md](depth-pruning.md).
2. **Zero-shot mask test (training-free, local):** zero the lowest-importance heads/channels
   at keep-75% / keep-50%, measure macro-F1 drop → the accuracy-vs-width curve before any
   recovery.
3. **Recovery-FT** of the surviving keep-fractions vs a from-scratch anchor → real GPU box.

## Expected
granite FFN is thin (18.7% params) and attention is 16.7%, so like low-rank the *mass* is
small; width pruning is mainly a **speed** lever (dense GEMM shrink), not size. Value depends
on whether depth alone gives enough speed — depth came out modest (keep-18 ≈ 18%), so width
may be worth stacking if the speed score binds.

Code: new (activation-magnitude channel/head ranking) — not yet written.
