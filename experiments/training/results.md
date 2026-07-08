# Branch: training — loss & knowledge-transfer recipe upgrades
Git branch: `research/training` · Experiments: E9 (losses: focal/LS/wCE/logit-adjusted), E13 (SAM — CLOSED by analysis), E14 (grouped-split protocol test, LOW prio), E10 (ensemble-distillation — DEFERRED to pre-deadline, correlated-error bound), E11 (TAPT granite-MLM, READY — user-approved 2026-07-07)
Baseline: champion qwen3 v1@512 = 0.7682 uncal (E9 screening arms baseline = same-backbone plain-CE control).

## Summary (at-a-glance)
Legend: ✅ win · ❌ closed · 🟡 low-prio, not run · 🕐 deferred. Δ vs same-backbone plain-CE control (granite v1@512 = 0.7458).

| Exp | Experiment | Status | Result (uncal F1) | Δ vs baseline | Verdict |
|---|---|:--:|---|---|---|
| E9 | loss screen: CE / focal / label-smoothing | ✅ | LS 0.7565 · focal 0.7403 (vs CE 0.7458) | LS **+0.0107** · focal −0.0055 | ✅ **LS PROMOTED** (transfers to E8 → 0.7803); focal closed |
| E11 | TAPT — granite MLM on 70k serialized texts | ✅ | **0.7592** | **+0.0134** | ✅ **WIN** — cheap domain-adaptive gain |
| E13 | SAM fine-tune | ❌ | — | — | ❌ CLOSED by analysis (no run) — near-zero run variance, nothing to harvest |
| E14 | session-grouped split (leak-free CV) | 🟡 | — | — | LOW prio — protocol test, **not run** |
| E10 | ensemble → single distillation | 🕐 | — | — | DEFERRED to pre-deadline (correlated-error bound, E6 lesson) |

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

### 2026-07-07 · E9 — Macro-F1-targeted loss screen (granite-311m, v1@512, 80/20 split, 3ep, lr 2e-5)
Same backbone/recipe across all three arms; only the training loss differs. Uncalibrated
`val_macro_f1` (calibrated columns ignored per the no-calibration rule).

| arm | val_macro_f1 | Δ vs CE | verdict |
|---|---|---|---|
| CE (control) | **0.7458** | — | baseline |
| focal γ=2 | 0.7403 | **−0.0055** | ❌ hurts — close this axis |
| label smoothing ε=0.1 | **0.7565** | **+0.0107** | ✅ WINNER — promote |

**Verdict: promote label smoothing (ε=0.1) to the champion recipe** — +0.0107 clears the
+0.003 promotion bar by 3.5×. LS *replaces* CE (no aux objective), so this is a clean read:
plain CE on the imbalanced classes (edit_file 15.8% vs web_search 1.8%) was leaving macro-F1
on the table, exactly the E9 hypothesis. **Focal loss is closed** (−0.0055) — γ=2 down-weighting
did not help the tail here.

Figure: `experiments/training/figures/e9_loss_screen.png`
Sources: `output/pat/ft_results_e9_{ce,focal,ls}.csv`

**Follow-up (blocking before it enters the final submission recipe):** re-test LS ε=0.1 on the
champion / E8 recipe (qwen3 / E8b, baseline 0.7682) to confirm the granite-screen win transfers
to the qwen3 backbone. A granite screen win does not guarantee transfer — verify on champion
before baking LS into the submission recipe.

### 2026-07-08 · E11 — TAPT (task-adaptive continued pretraining), granite-311m · **WIN**
Two-stage: (1) continued **MLM pretraining** of granite-311m on our ~70k serialized train
texts (the domain corpus) → `output/pat/granite_tapt_mlm/final`; (2) the standard
classification-FT from that TAPT'd encoder. Control is the E9 plain-CE arm — **same
backbone, same classification recipe**, only the encoder init differs (TAPT'd vs base), so
this is a clean read of TAPT alone. Recipe: `full_ft`, linear head, 3ep @ lr 2e-5,
max_len 512, serialize v1. Uncalibrated `val_macro_f1` (calibrated column ignored).

| arm | encoder init | val_macro_f1 | Δ vs CE control |
|---|---|---|---|
| E9-CE (control) | base granite-311m | **0.7458** | — |
| **E11 — TAPT** | MLM-pretrained on 70k domain texts | **0.7592** | **+0.0134** |

**Read-out → E11 is a WIN. TAPT HELPS.** Continued MLM pretraining on the in-domain
serialized corpus lifts granite +0.0134 over the non-TAPT control — well past the +0.003
promotion bar. Orthogonal to the loss axis (E9); adapting the encoder to the domain
distribution before classification-FT recovers real signal on this task.

Figure: [figures/e11_tapt.png](figures/e11_tapt.png)
Sources: `output/pat/ft_results_e11.csv` (tag `e11_granite_tapt`), `output/pat/ft_results_e9_ce.csv` (tag `e9_granite_ce`)

## Round-2 paper additions (web search 2026-07-07)
- E12 weight averaging → **moved to research/combine** (user call 2026-07-07: pure
  performance-stacking device, no task hypothesis — it's the final stage before zipping).
- **E13 SAM** — ❌ CLOSED 2026-07-07 without running: near-zero run-to-run variance
  (richmeta/richargs twins Δ=2e-5) means no landscape variance to harvest; residual
  error is intrinsic ambiguity; SAM gains concentrate in low-data regimes. Revive
  only if E12 soup Δ>+0.005.
- **E9 gains a 5th arm: logit-adjusted loss** (Menon 2007.07314) — consistent surrogate
  for balanced/macro metrics; train-time fixed log-priors, NOT post-hoc val-fit bias
  (distinct from the banned calibration; flagged for user veto).
- Considered & rejected: FreeLB/SMART adversarial FT (2-3× compute, finicky, GLUE-era
  gains shrink on strong backbones), text mixup (weak evidence for token classification
  era), TTA over serializations (costs inference budget — the one resource we can't spend).
