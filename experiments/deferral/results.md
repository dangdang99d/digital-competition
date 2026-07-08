# Branch: deferral — route low-confidence samples to a better source
Git branch: `research/deferral` · Baseline: **single-model champion qwen3 = 0.7682 uncal**. All scores uncalibrated macro-F1.

## Summary
Legend: ❌ failed · 🚫 prohibited.

| Exp | Experiment | Status | Result | Verdict |
|---|---|:--:|---|---|
| E6 | two-model per-class-τ fallback | ❌ | +0.0047 overall, but −0.0033 au / −0.0016 first-step | FAILS robustness — DROP |
| E7 | session-lookup deferral | 🚫 | — | DO NOT ATTEMPT (rules risk) |

## E6 — two-model per-class-τ fallback
- **What:** use qwen3, but on low-confidence cases defer to a second model.
- **Baseline:** single-model qwen3 = 0.7682.
- **Change:** add a hist0 fallback model + per-class thresholds τ_c; where qwen3's top1−top2 gap < τ_c, take hist0's argmax (τ_c greedy-fit).
- **Result:** +0.0047 overall (reproduces the proposal), but under honest 2-fold it beats base on only **2 of 4** robustness slices — held-out **au −0.0033** and **first-step −0.0016** (the two stress tests that matter). Also needs both models' weights → 2× inference.
- **Verdict:** ❌ **FAILS robustness — DROP.** The headline gain is a majority-regime (sim/later-step) artifact.

## E7 — session-lookup deferral
- **What:** exploit train/test session overlap to look up answers.
- **Verdict:** 🚫 **DO NOT ATTEMPT** — competition-rules risk (user decision 2026-07-07). Data property noted in memory as background only.

## Notes
- Margin (top1−top2, uncal) → correctness AUROC: hist0 0.832 / qwen3 0.848; wrong predictions pile at gap<1. Flip-to-top2 and pair-flip are dead; oracle two-model ceiling = 0.810.
- Figure: `figures/e6_robustness.png`.
