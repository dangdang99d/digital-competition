# Branch: combine — maximize LB by stacking every proven win
Branch: `kyusang_kvprune_svd` · logical group `research/combine` (the per-branch topology was planned but never created — all work is committed on kyusang_kvprune_svd) · Baseline: **champion-qwen3 on the matched 3,500 held-out slice = 0.7620 uncal** (LB SOTA = 0.77427). All scores uncalibrated macro-F1; matched-slice = the untouched 25% that `--full_data` folds OUT of training.

## Summary
Legend: 🏆 best · ✅ done · ⚠️ inconclusive.

| Exp | Arm / recipe | Model | Status | Result | Δ | Verdict |
|---|---|---|:--:|---|---|---|
| E8a | granite + richargs + full_data | granite | ✅ | 0.7706 | +0.0086 | WIN — beats champion, fast backbone |
| E8b | qwen3 + richargs + full_data | qwen3 | ✅ | 0.7643 | +0.0023 | weaker, near budget cliff |
| E8a+LS | E8a + label-smoothing ε=0.1 | granite | ✅ | **0.7803** (3.5k CV) → **LB 0.77738** | +0.0097 vs E8a | strong #2 on LB (fast, 5:06) |
| E8b+LS | E8b + label-smoothing ε=0.1 | qwen3 | 🏆 | **0.7659** (3.5k CV) → **LB 0.77921** | +0.0016 vs E8b | **🥇 LB SOTA — CV under-ranked it** |
| E12 | model soup, 3 seeds | granite | ⚠️ | 0.7803/0.7758/0.7628 | n/a | BOTCHED — each on own slice; soup never computed |

## E8a — granite + richargs + full_data
- **Model:** granite-311m.
- **What:** replicate the LB-SOTA recipe (granite, all-data) and add our richargs serialization axis.
- **Baseline:** champion-qwen3 on matched 3,500 slice = 0.7620.
- **Change:** backbone granite-311m + richargs serialization + `--full_data` (+75% of val into train), no calibration.
- **Result:** **0.7706** = **+0.0086**. Fast backbone (5:10, huge budget headroom).
- **Verdict:** ✅ WIN.

## E8b — qwen3 + richargs + full_data
- **Model:** qwen3-0.6B.
- **What:** same recipe on the best historical backbone.
- **Baseline:** same 0.7620.
- **Change:** backbone qwen3-0.6B + richargs + `--full_data`.
- **Result:** **0.7643** = **+0.0023**. Near the 9:06/10:00 inference budget cliff.
- **Verdict:** ✅ weaker than granite on the 3.5k CV slice — **but this inverted on LB exactly as predicted**: +LS (E8b+LS) → **LB 0.77921**, beating granite's 0.77738. qwen3's backbone edge is real; the 3.5k-slice CV just couldn't see it.

## E8a+LS — granite + richargs + full_data + label smoothing
- **Model:** granite-311m.
- **What:** stack the E9 label-smoothing win onto E8a.
- **Baseline:** E8a (CE) = 0.7706.
- **Change:** `--loss ls` (ε=0.1) replacing plain CE.
- **Result:** **0.7803** (full_data CV) = **+0.0097** vs E8a. (full_data CV, not the 14k val — high variance, don't over-read; the earlier "+0.006 above SOTA" framing is dropped.)
- **LB (submitted 07-08):** **0.77738** at 5:06 → beats prior SOTA 0.77427, but **loses to qwen3_ls's 0.77921** (E8b+LS). granite's full_data-CV lead reversed on LB.
- **Verdict:** ✅ strong #2 (fast, 5:06). Zip `submit_0707_granite_ls.zip` submitted (fp32 846M; parity-verified fp16 twin 539M available).

## E12 — model soup (3 granite+LS seeds → weight average)
- **Model:** granite-311m (E8a+LS recipe, seeds 42/43/44).
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
