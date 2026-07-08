# Branch: training — loss & knowledge-transfer recipe upgrades
Git branch: `research/training` · Baseline: **same-backbone plain-CE control, granite v1@512 = 0.7458 uncal**. All scores uncalibrated macro-F1.

## Summary
Legend: ✅ win · ❌ closed · 🟡 low-prio, not run · 🕐 deferred.

| Exp | Experiment | Model | Status | Result | Δ vs baseline | Verdict |
|---|---|---|:--:|---|---|---|
| E9 | loss screen: CE / focal / LS | granite | ✅ | LS 0.7565 · focal 0.7403 | LS **+0.0107** · focal −0.0055 | LS PROMOTED; focal closed |
| E11 | TAPT (MLM on 70k texts) | granite | ✅ | **0.7592** | **+0.0134** | WIN — cheap domain gain |
| E13 | SAM fine-tune | granite (planned) | ❌ | — | — | CLOSED by analysis (no run) |
| E14 | session-grouped split | qwen3 | 🟡 | — | — | low prio, not run |
| E10 | ensemble → single distillation | granite (planned) | 🕐 | — | — | deferred to pre-deadline |

## E9 — macro-F1-targeted loss screen
- **Model:** granite-311m.
- **What:** swap the training loss to something friendlier to imbalanced classes (edit_file 15.8% vs web_search 1.8%).
- **Baseline:** plain CE = 0.7458 (same backbone/recipe).
- **Change:** focal γ=2, or label-smoothing ε=0.1 — each *replaces* CE (no aux objective, clean read).
- **Result:** LS **0.7565 (+0.0107)**; focal 0.7403 (**−0.0055**).
- **Verdict:** ✅ **LS PROMOTED** (clears the +0.003 bar 3.5×; later transfers to E8 → 0.7803). Focal closed.

## E11 — TAPT (task-adaptive continued pretraining)
- **Model:** granite-311m.
- **What:** adapt the encoder to the domain before classification-FT.
- **Baseline:** E9 CE control = 0.7458 (base encoder init).
- **Change:** encoder init = granite continued-MLM-pretrained on our 70k serialized train texts; identical classification-FT after.
- **Result:** **0.7592** = **+0.0134**.
- **Verdict:** ✅ **WIN** — orthogonal to the loss axis; cheap domain-adaptive gain, candidate for the final recipe.

## E13 — SAM fine-tune
- **Model:** granite-311m (planned).
- **What:** sharpness-aware minimization for flatter, more generalizable minima.
- **Baseline:** best single (n/a — not run).
- **Change:** SAM optimizer wrapper.
- **Result:** — **not run**.
- **Verdict:** ❌ **CLOSED by analysis** — near-zero run-to-run variance (richmeta/richargs twins Δ=2e-5) means no landscape variance to harvest; error is intrinsic ambiguity. Revive only if E12 soup Δ>+0.005.

## E14 — session-grouped split (leak-free CV)
- **Model:** qwen3-0.6B (original recipe).
- **What:** a CV protocol with no session overlap between train/val (honest generalization estimate).
- **Baseline:** interleaved-val qwen3 = 0.7682 (vs known LB 0.76698).
- **Change:** group the split by session id.
- **Result:** — **not run**.
- **Verdict:** 🟡 LOW prio — protocol test, one grouped retrain.

## E10 — ensemble → single distillation
- **Model:** granite-311m student (planned).
- **What:** move 2-model ensemble knowledge into ONE model at 1× inference (`--distill_from` teacher logits).
- **Baseline:** best single model at the time.
- **Change:** train student on teacher-ensemble logits.
- **Result:** — **not run**.
- **Verdict:** 🕐 DEFERRED to pre-deadline — ensemble ceiling is correlated-error-bounded (E6 lesson); attempt only in the final days.
