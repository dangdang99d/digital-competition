# Branch: translation — ko→en pipeline for token surgery + English-specialist backbones
Git branch: `research/translation` · Experiments: E15a-d
Context: teammate's −0.17 result was INVALID (train/test language mismatch + translator
inconsistency + truncation — see teammate_work/miseo_koen analysis, 2026-07-07). The
hypothesis itself — English text enables token surgery & English-specialist models —
is untested. This branch tests it properly.

## Summary (at-a-glance)
Legend: ❌ no-go · ⛔ gated, not run. Budget = 10 min for 30k rows (must also fit the classifier).

| Exp | Experiment | Status | Result | Verdict |
|---|---|:--:|---|---|
| E15a | NLLB-600M qualification (throughput/quality/preservation) | ❌ | **93 min / 30k** (~31 min even @3× speedup) ≫ 10-min budget | ❌ **NO-GO (throughput)** — translation-at-inference can't fit; kills the line |
| E15b | code-span protection (mask→translate→restore) | ⛔ | — | Gated on E15a → not run |
| E15c | EN-data classifiers vs KO baselines (**THE payoff test**) | ⛔ | — | Gated on E15a/b — **the core hypothesis (does EN input help?) was never tested**, blocked by E15a's throughput NO-GO |
| E15d | translator compression + 30k budget fit | ⛔ | — | Gated on E15c → not run |

> ⚠️ **Open, not disproven:** E15a killed translation *at inference* on throughput, but the payoff question (E15c: do EN-data / EN-specialist classifiers beat KO baselines?) is independent of inference cost — you could translate the TRAIN set offline and only ship a KO/EN classifier. That path was never run.

## E15a · NLLB-600M qualification (GO/NO-GO — no tournament)
**Simplified per user decision 2026-07-07:** MT quality scales with size (unlike our
classifier task), so skip the multi-model bake-off — take the largest deployable,
**facebook/nllb-200-distilled-600M**, as the presumptive translator and qualify it:
1. **Quality:** CometKiwi (Unbabel/wmt22-cometkiwi-da) on 300 stratified Korean train
   samples + 30-pair human sample sheet → user review
2. **Code preservation:** identifier/path/number exact-match rate, WITH and WITHOUT
   the E15b protection layer (preservation is ~size-independent — must be measured)
3. **Throughput:** sent/s @ fp16/3090 → projected 30k-load translation share (bar: fits
   alongside classifier in 10 min)
⚠ **License: CC-BY-NC — user must confirm competition eligibility.** Licensed fallback
if it fails: facebook/m2m100_418M (MIT) — only measured if needed. All other candidates
(opus-mt floor, 1.3B, 3B/2.4B reference rows) DROPPED.
**Note:** translator choice bakes into training data (E15c retrains on its output);
swapping later = retranslate + retrain. Accepted.

## E15b · Code-span protection pipeline
Mask code/paths/identifiers → <C0>,<C1>… → translate → restore. Re-measure preservation
(target ~100%) + CometKiwi delta. Small script, reusable at train and inference time.

## E15c · The real test: EN-data classifiers vs KO baselines (the payoff experiment)
Translate ALL train (chosen translator + protection; consistent everywhere). Train,
same recipe/split/uncal protocol:
  (i) qwen3 on EN data            vs qwen3 on KO (0.7682)      → "does EN input help multilingual?"
  (ii) granite on EN data         vs granite on KO             → same, SOTA backbone
  (iii) **English-only** backbone on EN data — microsoft/deberta-v3-large (304M) and/or
        answerdotai/ModernBERT-large (395M, 8k ctx) vs (i)/(ii) → "do EN-specialists beat
        multilingual, given EN input?" (the user's question — capacity not spread over
        100 languages, deeper EN pretraining)
**Read-out:** any EN arm > its KO baseline +0.003 → pursue + unlock token-surgery
variants (number bucketing / synonym-collapse INSIDE the EN text); all ≤ → close branch
with real numbers. Prediction on record (weak-semantic labels → translation collapses
class-bearing surface forms): skeptical, but this time the test is valid.

## E15d · Translator compression + budget fit — MANDATORY for any pick except opus-mt
(1 GB zip cap: only opus-mt fits uncompressed next to a granite classifier)
- Vocab-prune the translator (NLLB vocab 256k — embedding-heavy like bge was; our
  proven prune_vocab technique applies: keep tokens seen in KO train text + EN outputs)
- fp16; optional depth-prune encoder/decoder; measure ΔCometKiwi + Δpreservation +
  throughput gain; end-to-end 30k-load timing with classifier in sequence.

## Deployment size math (user-proposed target combo, 2026-07-07)
Both-pruned path: qwen3 vocab+depth(28→14) ≈ 0.55GB + NLLB-600M vocab(256k→30k)+
depth(12+12→6+6) ≈ 0.41GB → **~0.96GB — inside 1GB, zero margin**. Time ≈ 4:30-5:00
classify + 1-2.5min translate ≈ 6-8min. Gated: E15c win (does EN help at all?) →
E16 (qwen3 depth) + E15d (NLLB compression) → combo zip.
**Fallback combos with margin:** granite-fp16 0.31 + NLLB-pruned 0.41 = **0.72GB**;
DeBERTa-v3-base 0.37 + NLLB-pruned = 0.78GB. If qwen3-EN ≈ granite-EN in E15c, take
the margin. E17 token-budget batching widens the time margin for all combos.

## Results
(append per sub-experiment)
