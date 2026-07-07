# Branch: training — loss & knowledge-transfer recipe upgrades
Git branch: `research/training` · Experiments: E9 (losses: focal/LS/wCE/logit-adjusted), E12 (soup/SWA), E13 (SAM, backlog), E10 (ensemble-distillation — DEFERRED to pre-deadline, correlated-error bound), E11 (TAPT granite-MLM, READY — user-approved 2026-07-07)
Baseline: champion qwen3 v1@512 = 0.7682 uncal (E9 screening arms baseline = same-backbone plain-CE control).

## Why these (papers/ + local evidence)
- **E9 losses:** metric is macro-F1 but training is plain CE on imbalanced classes
  (edit_file 15.8% vs web_search 1.8%). FocalLoss (ICCV'17) + LabelSmoothing (NeurIPS'19)
  are the canonical levers; never tried here. Teammate's class-weighted CE ran only
  inside a LoRA recipe (0.74093 LB) — confounded, not a clean read. Cheap to screen.
- **E10 distillation:** the strongest idea. Oracle two-model ceiling = 0.810 vs qwen3
  0.7682, but E6 proved post-hoc routing can't reach it and 2× inference is unaffordable.
  Distillation (DistilBERT/MiniLM lineage) is the *legitimate* way to move ensemble
  knowledge into ONE model at 1× inference. Local evidence it works on this task:
  teammate's granite distillation = CV 0.754 vs 0.7461 plain FT (+0.008). `--distill_from`
  plumbing already exists in src.finetune (teacher-logits npz).
- **E11 TAPT (backlog):** TODBERT/IntentBERT-style task-adaptive continued pretraining
  on our 70k serialized texts before FT. Orthogonal to everything; occasionally +1pt.
- **Rejected from papers/ (documented so we don't revisit):** CPFT/SetFit/DNNC/protonets
  (few-shot methods; we have 70k and RevisitFewShot says plain FT wins), ADB/OOD suite
  (no abstain class in this comp), Q8BERT-style int8 (LB timeout precedent), morpheme
  tokenizer swap (too invasive), more contrastive aux (supcon already −0.7pt here).

## Results
(append: date · experiment · arm · Δ vs baseline · figure · commit)

## Round-2 paper additions (web search 2026-07-07)
- **E12 weight averaging** — Model Soups (Wortsman ICML'22) / SWA (Izmailov'18): free
  accuracy from averaging same-recipe weights; zero inference cost. Piggybacks on E9/E11.
- **E13 SAM** — Bahri et al. 2110.08529: flat-minima optimizer, large LM-FT gains at
  +25-100% train compute; gated behind E12 (same mechanism family).
- **E9 gains a 5th arm: logit-adjusted loss** (Menon 2007.07314) — consistent surrogate
  for balanced/macro metrics; train-time fixed log-priors, NOT post-hoc val-fit bias
  (distinct from the banned calibration; flagged for user veto).
- Considered & rejected: FreeLB/SMART adversarial FT (2-3× compute, finicky, GLUE-era
  gains shrink on strong backbones), text mixup (weak evidence for token classification
  era), TTA over serializations (costs inference budget — the one resource we can't spend).
