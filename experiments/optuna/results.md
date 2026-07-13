# Branch: optuna (E28) — hyperparameter search for performance maximization

Logical group `research/optuna`. Ran on a rented **vast.ai 4× RTX 3090** instance (the ocean
fleet was unavailable). All scores **raw uncalibrated macro-F1** (project invariant: NO logit
calibration).

Objective: squeeze more single-model performance out of the champion recipe by tuning training
hyperparameters jointly with optuna (TPE), instead of the one-axis-at-a-time screens (E9/E25).
Winners keep `--init_seed 42` so they stay soup/ensemble-compatible ([[model-soup-needs-shared-init]]).

**Baselines:**
- champion granite E8a `richargs + LS ε=0.1 + full_data + bf16`: LB **0.77738** (single-model)
- other single-model LB: granite+TTA 0.77931 · qwen3_ls 0.77921
- ensemble LB (context, not the E28 target): E26-C trio **0.78719**
- **in-search anchor** (champion config, same fold/harness, best-epoch): **0.7586** — see Results

## Design

- **Target:** the champion granite recipe (richargs + LS + bf16), tuned around it.
- **Search objective:** macro-F1 on the **session-grouped fold-0 holdout** (leakage-free
  `session_fold_indices`), NOT the row-stratified/3.5k slice — those mis-rank (E8 CV→LB reversal).
  This is the value optuna maximizes; **LB remains the judge** for any promoted model.
  Reported F1 = the run's **best-epoch** value (finetune.py RAM-snapshot). ⚠️ **This choice of
  objective drives the promotion caveat below** — the search rewards the best epoch, so promotion
  must too.
- **Fixed** (settled by prior experiments): granite-311m · richargs · `--loss ls` ·
  `--init_seed 42` · max_len 512 · bf16-auto · grad-ckpt auto-off · E19 fast shape
  (real batch, `--grad_accum 1`, `--group_by_length`).
- **Search space:** 6 params, table below.

## Search space (round 1)

`selected` = best trial (**t043**, fold-0 F1 0.7724). `importance` = optuna fANOVA over the 58
completed trials.

| parameter | scale | range | champion | **selected (t043)** | importance |
|---|---|---|---|:--:|:--:|
| `lr` | log-uniform | 5e-6 → 5e-5 | 2e-5 | **2.60e-5** | **0.50** |
| `eff_batch` | categorical | {8, 16, 32} | 16 | **8** | 0.21 |
| `epochs` | int | 2 → 5 | 3 | **5** ⚠️ | 0.19 |
| `warmup_ratio` | uniform | 0.0 → 0.15 | 0.05 | **0.103** | 0.04 |
| `label_smoothing` | uniform | 0.02 → 0.20 | 0.10 | **0.187** | 0.04 |
| `weight_decay` | log-uniform | 1e-3 → 0.1 | 0.01 | **0.057** | 0.02 |

**Read-out:** `lr` dominates (half the explained variance); the top tier converges on
**lr ≈ 3–5e-5 (2–2.5× the champion's 2e-5), eff_batch 8, heavy smoothing ε ≈ 0.12–0.20** — a
distinctly higher-lr / smaller-batch / stronger-regularization corner than the champion recipe.
⚠️ `epochs=5` is an **artifact of the best-epoch objective** — see the promotion caveat.

## Results

**Search:** 60 trials (4 workers × 15), 58 complete + 23 pruned (MedianPruner, 8 startup). Best
= **t043 = 0.7724**, champion anchor **t024 = 0.7586** → **+0.0138 on the same fold**. Nearly the
whole top-8 clears the anchor by ≥ +0.009, so the gain is broad, not a single lucky trial.

Top-8 (fold-0 best-epoch F1):

| trial | F1 | lr | ep | eb | wu | ε | wd |
|---|---|---|---|---|---|---|---|
| **t043** | **0.7724** | 2.60e-5 | 5 | 8 | 0.103 | 0.187 | 0.057 |
| t048 | 0.7715 | 3.52e-5 | 5 | 8 | 0.037 | 0.174 | 0.059 |
| t023 | 0.7709 | 4.35e-5 | 5 | 8 | 0.077 | 0.116 | 0.010 |
| t041 | 0.7707 | 5.00e-5 | 5 | 8 | 0.052 | 0.186 | 0.060 |
| t033 | 0.7701 | 4.12e-5 | 5 | 8 | 0.073 | 0.195 | 0.022 |
| t051 | 0.7700 | 3.06e-5 | 5 | 16 | 0.076 | 0.199 | 0.062 |
| t017 | 0.7682 | 4.46e-5 | 4 | 16 | 0.079 | 0.022 | 0.004 |
| t019 | 0.7684 | 1.62e-5 | 5 | 8 | 0.135 | 0.150 | 0.007 |

**⚠️ PROMOTION CAVEAT — the search objective and the promotion recipe must agree on epoch
selection.** Every top config's per-epoch curve **peaks at epoch 3, then declines**:

| trial | ep1 | ep2 | **ep3 (peak)** | ep4 | ep5 (last) |
|---|---|---|---|---|---|
| t043 | 0.724 | 0.764 | **0.772** | 0.756 | 0.739 |
| t048 | 0.720 | 0.756 | **0.772** | 0.762 | 0.747 |
| t023 | 0.708 | 0.754 | **0.771** | 0.763 | 0.748 |

Because the objective reports **best-epoch** F1, extra epochs were free — `epochs=5` scored the
same as `epochs=3` and TPE drifted upward by noise; `epochs` is effectively meaningless ≥3.

The first promotion attempt shipped these configs via **`--all_data`** (all 70k, no holdout →
**last epoch** saved). That ships the **epoch-5 overfit tail** (~0.739 fold-equiv), not the
epoch-3 peak (~0.772). **LB confirmed the failure: `submit_0713_g_t043.zip` (all_data t043) →
0.75934** — a −0.018 regression vs champion, exactly the ep5 curve value plus the champion's
fold→LB offset. The model is not broken; it is two epochs past its optimum.

**Fix (CONFIRMED):** re-promoted via **`--full_data`** (66.5k train + 3.5k eval → best-epoch
RAM snapshot restored, the recipe that built champion LB 0.77738). Four retrains banked:
`e28_full_{t043,t048,t023,t017}`. **`submit_0713_g_t043fd.zip` → LB 0.78155 (5:04) = NEW
single-model SOTA** (+0.0042 vs granite_ls 0.77738, +0.0022 vs TTA 0.77931, past qwen3_ls
0.77921). The full_data fix recovered +0.022 of LB over the broken all_data zip (0.75934).

**Fold vs slice — the fold won.** The 3.5k row-stratified slice ranked all four E28 configs
BELOW champion (0.7803) and put t017 (0.7750) above t043 (0.7705); the leak-free session-fold
ranked t043 highest and beating champion. **LB agreed with the fold** (t043fd 0.78155 > champion
0.77738), confirming — again, per [[e8-granite-beats-qwen3-cv]] — that the 3.5k slice mis-ranks
and the session-grouped fold is the trustworthy local signal.

**Methodological takeaway (generalizes past E28):** when the HPO objective is best-epoch, the
final-model recipe must also select the best epoch. Blind last-epoch promotion (`--all_data`)
silently ships the overfit tail. Prefer `--full_data` (keeps epoch selection) unless a separate
early-stopping signal is wired into the all-data path.

### Ensemble (14k session-fold-0 screen)

To try to beat the E26-C trio SOTA (LB 0.78719), the 7 kept **search checkpoints** (session-fold-0
trained) were harvested on the **fold-0 holdout (14,000 rows, leak-free** for them — 4× the 3.5k
slice E26 was limited to; `experiments/optuna/screen_e28_foldval.py`). Brute-force over all subsets,
uniform softmax mean:

| level | best combo | fold-0 (14k) F1 | Δ vs best single |
|---|---|---|---|
| single | t043 | 0.7719 | — |
| pair | t019+t023 | 0.7758 | +0.0039 |
| **trio** | **t019+t023+t043** | **0.7774** | **+0.0055** |
| quad | +t011/t007 | 0.7767 | *worse (4th member hurts)* |

**Diversity is learning-rate spread only** (t019 1.6e-5 / t043 2.6e-5 / t023 4.35e-5) — same
backbone/serialization/data/init_seed, so weaker error-decorrelation than E26's serialization+data
diversity (which gave +0.008). The +0.0055 here is real but modest; the losing quad confirms the
low-diversity ceiling. Screen used session-fold checkpoints; **deployment uses `--full_data`
members** (t019 retrained on sandbox, val 0.7754) — standard screen-then-retrain transfer.

Built `submit_0713_e28_trio.zip` (842M; vocab-prune union 54,688 = same keep-set as E26; fp16;
uniform mean via `script_ensemble.py`; **parity 0/1000 flips** all 3 members; smoke PASS).
**🥇 LB 0.78780 (6:39) = NEW OVERALL SOTA** (+0.00061 vs E26-C 0.78719). The trio gained
**+0.00625 over its best single** (t043fd 0.78155) — almost exactly the +0.0055 fold-screen
prediction, so **the 14k leak-free screen was a reliable LB predictor**. Even lr-spread-only
diversity, on a stronger E28 base, edges the serialization-diverse E26-C trio.

## Plan

| Phase | What | Status |
|---|---|:--:|
| 0 | design: session-grouped fold-0 objective, 6-param space | ✅ |
| 1 | scaffold `src/optuna_search.py` (subprocess objective around `src.finetune`, shared SQLite TPE study, per-epoch MedianPruner, cuInit guard, running-best weight retention) + `--weight_decay` / `--all_data` flags in finetune.py | ✅ |
| 2 | search: 60 trials on vast 4×3090; best t043 0.7724 vs anchor 0.7586 (+0.0138) | ✅ |
| 3 | promote top-k from scratch. **v1 `--all_data` → FAILED** (last-epoch tail; t043 LB 0.75934). **v2 `--full_data`** (best-epoch): t043fd **LB 0.78155 = single-model SOTA** ✅ | ✅ |
| 4 | zips: `submit_0713_g_t043fd.zip` (single **LB 0.78155**) + `submit_0713_e28_trio.zip` (trio **LB 0.78780 = NEW SOTA**, +0.00061 vs E26-C) | ✅ **DONE — E28 trio is the new overall SOTA** |

## Log

- **2026-07-11 → 07-12** — scaffold built; search space finalized to the E19 fast shape
  (`--grad_accum` explicit, `--group_by_length`); ran 60 TPE trials across 4× 3090 on vast.
- **2026-07-12** — enqueued the champion config as an in-study anchor (t024 = 0.7586) so every
  result reads as a same-fold Δ, not an extrapolation.
- **2026-07-13** — best t043 = 0.7724 (+0.0138 vs anchor). **`--all_data` promotion mis-ships the
  epoch-5 tail** → t043 LB **0.75934** (−0.018). Pivot to **`--full_data`** (best-epoch selection);
  re-promoting t043/t048/t023/t017. See the promotion caveat above.

## Artifacts

- Study DB + all trial logs + worker CSVs + kept checkpoints on NFS: `output/optuna/vast_r1/`
  (`e28.db`, `logs/`, `trials/`, `final/` = all_data v1, `final_fd/` = full_data v2).
- Reproduce any trial: `finetune.py --tag e28_tNNN` with its params from `e28.db`.
