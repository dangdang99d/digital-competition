# Branch: training — loss & knowledge-transfer recipe upgrades
Branch: `kyusang_kvprune_svd` · logical group `research/training` (the per-branch topology was planned but never created — all work is committed on kyusang_kvprune_svd) · Baseline: **same-backbone plain-CE control, granite v1@512 = 0.7458 uncal**. All scores uncalibrated macro-F1.

## Summary
Legend: ✅ win · ❌ closed · 🟡 low-prio, not run · 🕐 deferred.

| Exp | Experiment | Model | Status | Result | Δ vs baseline | Verdict |
|---|---|---|:--:|---|---|---|
| E9 | loss screen: CE / focal / LS / wce / la | granite | ✅ | LS 0.7565 · wce 0.7541 · la 0.7456 · focal 0.7403 | LS **+0.0107** · wce **+0.0083** · la −0.0002 · focal −0.0055 | LS PROMOTED; wce clears bar (<LS); la & focal closed |
| E11 | TAPT (MLM on 56k train-slice, v1) | granite | ✅ | **0.7592** | **+0.0134** | WIN — cheap domain gain, no val leak |
| E13 | SAM fine-tune | granite (planned) | ❌ | — | — | CLOSED by analysis (no run) |
| E14 | session-grouped split | qwen3 | 🟡 | — | — | low prio, not run |
| E10 | ensemble → single distillation | granite (planned) | 🕐 | — | — | deferred to pre-deadline |

## E9 — macro-F1-targeted loss screen
- **Model:** granite-311m.
- **What:** swap the training loss to something friendlier to imbalanced classes (edit_file 15.8% vs web_search 1.8%).
- **Baseline:** plain CE = 0.7458 (same backbone/recipe).
- **Change:** each *replaces* CE (no aux objective, clean read). Five arms:
  - focal γ=2 — `-(1-p_t)^γ·log p_t`
  - LS ε=0.1 — `(1-ε)·CE + ε·CE(uniform)`
  - **wce** — class-weighted CE, 'balanced' inverse-freq weights `w_k = N/(K·count_k)`
  - **la** τ=1.0 — logit-adjusted, `CE(logits + log π)` (train priors → inference stays raw argmax)
- **Result** (uncal val macro-F1):

  | Loss | val_macro_f1 | Δ vs CE 0.7458 | Verdict |
  |---|---|---|---|
  | CE (control) | 0.7458 | — | baseline |
  | **LS ε=0.1** | **0.7565** | **+0.0107** | PROMOTED (winner) |
  | **wce** (balanced) | 0.7541 | +0.0083 | clears bar, but < LS |
  | la (τ=1.0) | 0.7456 | −0.0002 | flat → CLOSE |
  | focal γ=2 | 0.7403 | −0.0055 | CLOSE |
- **Verdict:** ✅ **LS PROMOTED** (clears the +0.003 bar 3.5×; later transfers to E8 → 0.7803). **The two losses actually *designed* for macro-F1 split: class-weighted CE helps (+0.0083, a legit #2) but logit-adjustment at τ=1 does nothing (−0.0002) — and neither beats label smoothing, which isn't macro-designed at all.** Fits the weak-semantic-labels theme: gains come from regularization, not rare-class reweighting. `wce`/`la` added to finetune.py 2026-07-08 (`--loss wce|la`, `--la_tau`); commands auto-logged in `sbatch/logs/e9-granite-{wce,la}.out`.
- **Follow-up (untaken):** la τ was fixed at 1.0 (the theoretically-consistent value); a τ∈{0.5,1.5,2} sweep could revisit, but LS already won.

## E11 — TAPT (task-adaptive continued pretraining)
- **Model:** granite-311m.
- **What:** adapt the encoder to the domain before classification-FT.
- **Baseline:** E9 CE control = 0.7458 (base encoder init).
- **Change:** encoder init = granite continued-MLM-pretrained on our **56k train-split** serialized texts (seed-42 train slice only; the 14k val was held OUT of MLM — no leakage), **v1 serialization** (matches the downstream classify format), mlm_prob 0.15, 2 epochs; identical classification-FT after. (Audit note: earlier docs said "70k richargs" — the actual run was 56k / v1, which is the correct leak-free choice.)
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
