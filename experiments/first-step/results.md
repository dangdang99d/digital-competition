# Branch: first-step — the zero-history failure mode
Git branch: `research/first-step` · Baseline: **generalist on the 1,807 first-step val slice = 0.555 uncal** (qwen3 0.573). All scores uncalibrated macro-F1 on the identical slice.

## Summary
Legend: ❌ specializing hurts.

| Exp (arm) | Status | Result | Δ vs generalist 0.555 | Verdict |
|---|:--:|---|---|---|
| E1 · `--zero_history` (7k real first-steps) | ✅ | 0.4283 | −0.127 | specializing HURTS |
| E1 · `--strip_history` (all 56k, history stripped) | ✅ | 0.4407 | −0.114 | specializing HURTS |

## E1 — first-step specialist ceiling (A/B)
- **What:** does a model *specialized* on first-steps (zero history) beat the generalist on that slice?
- **Baseline:** the generalist (history-intact, all-position training) = 0.555 on the 1,807 first-step slice.
- **Change:** two specialist arms — (a) train only on the 7k real first-steps (`--zero_history`); (b) train on all 56k with history stripped (`--strip_history`). Both bge-m3, v1@1024.
- **Result:** (a) **0.4283** (−0.127); (b) **0.4407** (−0.114). More data helps only +0.012 — the gap is training on history-intact samples, not data volume.
- **Verdict:** ❌ **First-step error is INTRINSIC.** Specializing costs ~0.11 vs the generalist → keep the single generalist encoder; **do NOT build a first-step / has-history branch.**

## Notes
- step==1 ⟺ len(history)==0 exactly (9,000/70,000). First-step error ~2× mid-session (36.7% vs ~18%); models prior-collapse onto list_directory/plan_task. Bigger model barely helps (+0.018).
- Figure: `figures/firststep_ceiling_ab.png`.
