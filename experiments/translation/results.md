# Branch: translation — ko→en pipeline for token surgery + English-specialist backbones
Git branch: `research/translation` · Baseline: **qwen3 KO = 0.7682 uncal**; inference budget = 10 min / 30k rows (must also fit the classifier). All scores uncalibrated macro-F1.

Hypothesis: English text enables token surgery and English-specialist models. (A teammate's −0.17 result was invalid — train/test language mismatch + translator inconsistency; this branch tests the idea properly.) Note: E15a/b/d models are the **translator**; the classifier models (granite/qwen3/EN-specialist) only enter at E15c.

## Summary
Legend: ❌ no-go · ⛔ gated, not run.

| Exp | Experiment | Model | Status | Result | Verdict |
|---|---|---|:--:|---|---|
| E15a | NLLB-600M qualification | NLLB-200-distilled-600M (translator) | ❌ | 93 min / 30k ≫ budget | NO-GO (throughput) |
| E15b | code-span protection | NLLB (translator) | ⛔ | — | gated on E15a |
| E15c | EN-data classifiers vs KO (**payoff test**) | qwen3 / granite / DeBERTa-v3 / ModernBERT | ⛔ | — | never tested |
| E15d | translator compression + budget fit | NLLB (translator) | ⛔ | — | gated on E15c |

## E15a — NLLB-600M qualification (GO/NO-GO)
- **Model:** `facebook/nllb-200-distilled-600M` (translator — not a classifier).
- **What:** qualify NLLB-600M as the ko→en translator (quality / code-preservation / throughput).
- **Baseline:** the 10-min / 30k inference budget (translator must fit alongside the classifier).
- **Change:** translate KO→EN at inference with NLLB-600M.
- **Result:** **93 min for 30k** (~31 min even at a 3× speedup) — far over budget.
- **Verdict:** ❌ **NO-GO (throughput)** — translation-at-inference can't fit; this kills the deploy path.

## E15b — code-span protection
- **Model:** NLLB translator (+ protection layer).
- **What:** mask code/paths/identifiers → translate → restore, to keep preservation ~100%.
- **Change:** protection layer around the translator.
- **Result:** — **not run** (gated on E15a).
- **Verdict:** ⛔.

## E15c — EN-data classifiers vs KO baselines (the payoff test)
- **Model:** classifiers — qwen3-0.6B, granite-311m, and EN-specialists DeBERTa-v3-large / ModernBERT-large (trained on EN-translated data).
- **What:** the actual "does English help?" question. (i) qwen3-EN vs qwen3-KO, (ii) granite-EN vs granite-KO, (iii) EN-only backbone vs the multilingual ones.
- **Baseline:** each backbone's KO twin (qwen3 0.7682, etc.).
- **Change:** EN-translated training data (+ optional EN-specialist backbone).
- **Result:** — **not run**.
- **Verdict:** ⛔ **core hypothesis never tested.** NOTE: this is independent of inference throughput — you can translate the TRAIN set offline and ship only a KO/EN classifier, so E15a's NO-GO does not have to block E15c.

## E15d — translator compression + budget fit
- **Model:** NLLB translator (vocab-pruned / fp16 / depth-pruned).
- **What:** shrink the translator to fit the 1 GB zip / 10-min budget.
- **Result:** — **not run** (gated on E15c).
- **Verdict:** ⛔.
