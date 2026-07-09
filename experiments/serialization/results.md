# Branch: serialization — input-format gains on the best backbone
Branch: `kyusang_kvprune_svd` · logical group `research/serialization` (the per-branch topology was planned but never created — all work is committed on kyusang_kvprune_svd) · Baseline: **qwen3 v1@512 = 0.7682 uncal**. All scores uncalibrated macro-F1.

## Summary
Legend: ⛔ gated, not run.

| Exp | Experiment | Model | Status | Result | Verdict |
|---|---|---|:--:|---|---|
| E2 | richargs × qwen3 (single-axis ablation) | qwen3 | ⛔ | — | correctly skipped (gated on E8b) |

## E2 — richargs × qwen3
- **Model:** qwen3-0.6B. (Prior serialization findings below were measured on bge-m3.)
- **What:** isolate the richargs serialization gain on the qwen3 backbone (attribution diagnostic).
- **Baseline:** qwen3 v1@512 = 0.7682.
- **Change:** serialize `richargs` (rich meta header + arg-path basenames) instead of v1.
- **Result:** — **not run**. Gated on E8b, which already carries richargs and landed fine (0.7643), so the isolated ablation wasn't needed.
- **Verdict:** ⛔ **correctly skipped.**

## Notes (prior, bge-m3)
- richmeta +0.0117 uncal (0.7498→0.7615); richargs ≈ richmeta on CV (+0.00002) but **> richmeta on LB** (the hidden test rewards arg-path stripping).
- All info-REMOVAL variants regress (nometa −0.022, leanact −0.028). Hypothesis: serialization and backbone gains are ~orthogonal.
