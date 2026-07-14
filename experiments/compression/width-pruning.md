# Width pruning (structured) — remove attention heads / FFN neurons + recovery FT

**Method:** rank units by importance (FLAP fluctuation, Wanda-sp = |w|·‖x‖ activation
magnitude) → drop low-importance attention heads and/or FFN channels → recovery FT. Output =
smaller *dense* model, real GEMM shrink, speedup unconditional on any HW. Index:
[results.md](results.md).

| Model | Status |
|---|---|
| granite | ✅ DONE 2026-07-15 — FFN-neuron width prune (ModernBERT GeGLU) + recovery; see below |
| qwen3 | 🔲 not run (qwen3 line closed) |

## granite — FFN-neuron width prune results (2026-07-15, honest 3.5k held-out, base 0.78578)

`_prune_ffn_modernbert` (added to finetune.py): score GeGLU intermediate neurons by
`|Wi_input|·|Wi_gate|·|Wo_col|`, keep top-k, slice `Wi`/`Wo`, update `intermediate_size`.
Produces a smaller **dense** FFN (`1152 → k` neurons) — no sparsity, T4-safe. + 2ep LS recovery.

| keep | I: 1152→ | size Δ (whole model) | net ΔF1 (recovered) |
|---|---|---|---|
| 75% | 864 | −4.7% | −0.0096 |
| 50% | 576 | −9.4% | −0.0032 |
| 25% | 288 | −14.0% | −0.0047 |

**Read-out:** width prune costs −0.003…−0.010; the **ordering is 3.5k noise** (keep-75%
worse than keep-50%/25% is not real — best-epoch variance, ±0.003). Standalone width is a
modest lever; **its real value is stacked with depth** — the depth-18 × ffn-75 combo (one
joint recovery) nets −0.0017 at 1.31× speed, the program winner ([results.md](results.md)
Stage-B). Head pruning (the attention-side width analogue) not done — `prune_attn_heads` is
BERT-style and needs a ModernBERT fused-Wqkv adaptation.

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
