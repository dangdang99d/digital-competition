# Branch: combine — maximize LB by stacking every proven win
Git branch: `research/combine` · Baseline: **champion-qwen3 on the matched 3,500 held-out slice = 0.7620 uncal** (LB SOTA = 0.77427). All scores uncalibrated macro-F1; matched-slice = the untouched 25% that `--full_data` folds OUT of training.

## Summary
Legend: 🏆 best · ✅ done · ⚠️ inconclusive.

| Exp | Arm / recipe | Status | Result | Δ | Verdict |
|---|---|:--:|---|---|---|
| E8a | granite + richargs + full_data | ✅ | 0.7706 | +0.0086 | WIN — beats champion, fast backbone |
| E8b | qwen3 + richargs + full_data | ✅ | 0.7643 | +0.0023 | weaker, near budget cliff |
| E8a+LS | E8a + label-smoothing ε=0.1 | 🏆 | **0.7803** | +0.0097 vs E8a | **BEST — top submission** |
| E12 | model soup, 3 granite+LS seeds | ⚠️ | 0.7803/0.7758/0.7628 | n/a | BOTCHED — re-run split-fixed |

## E8a — granite + richargs + full_data
- **What:** replicate the LB-SOTA recipe (granite, all-data) and add our richargs serialization axis.
- **Baseline:** champion-qwen3 on matched 3,500 slice = 0.7620.
- **Change:** backbone granite-311m + richargs serialization + `--full_data` (+75% of val into train), no calibration.
- **Result:** **0.7706** = **+0.0086**. Fast backbone (5:10, huge budget headroom).
- **Verdict:** ✅ WIN.

## E8b — qwen3 + richargs + full_data
- **What:** same recipe on the best historical backbone.
- **Baseline:** same 0.7620.
- **Change:** backbone qwen3-0.6B + richargs + `--full_data`.
- **Result:** **0.7643** = **+0.0023**. Near the 9:06/10:00 inference budget cliff.
- **Verdict:** ✅ weaker than granite on CV (may invert on LB via qwen3's backbone edge).

## E8a+LS — granite + richargs + full_data + label smoothing
- **What:** stack the E9 label-smoothing win onto E8a.
- **Baseline:** E8a (CE) = 0.7706.
- **Change:** `--loss ls` (ε=0.1) replacing plain CE.
- **Result:** **0.7803** = **+0.0097** vs E8a; **+0.0060 above LB SOTA on CV** (first arm to clear SOTA on its own slice).
- **Verdict:** 🏆 **BEST of session, top submission.** Zip `submit_0707_granite_ls.zip` built + verified (val reproduced 0.7799, parity 5/5, no logit_bias).

## E12 — model soup (3 granite+LS seeds → weight average)
- **What:** average the E8a+LS weights across seeds for a free bump.
- **Baseline:** best single seed of the same recipe.
- **Change:** train seeds 42/43/44, average the weights.
- **Result:** seeds 0.7803 / 0.7758 / 0.7628 — but `--seed` also reshuffles the train/val split, so **each seed trained and evaluated on a different slice** → not comparable, not soup-able.
- **Verdict:** ⚠️ **BOTCHED.** Re-run with the split held fixed (vary only init/shuffle seed) before any read.

## Notes (LB-proven axes this branch stacks)
- Backbone: qwen3 > bge-m3 > granite at equal recipe (but granite = fast + CV-best here).
- Serialization: richargs > richmeta **on LB** (0.76274 vs 0.76053) despite byte-identical CV — ship richargs.
- All-data training: +0.014 LB; full-data granite has no fold-CV↔LB gap.
- Budget cliff: int8 qwen3 timed out; fp16 qwen3 = 9:06/10:00. granite (5:10) has headroom.
- ⚠️ CV ≠ LB: deltas here are directional; the hidden test rewards things CV can't see.
