# Branch: performance-boost — performance program board (what actually improved the score)

Parallel to [compression/results.md](../compression/results.md): that board tracks **size↓ /
speed↑**; this one tracks **accuracy↑**. It is the curated shortlist — of the ~30 experiments
run, only the levers that **demonstrably raised macro-F1** (or are strong pending candidates)
are collected here, so the final-model recipe can be assembled from them without re-reading the
full [EXPERIMENTS.md](../EXPERIMENTS.md) dashboard.

**Selection rule for inclusion:** a lever earns a row only if it either (a) moved the **LB**
positively, or (b) moved the **from-scratch CV anchor** by ≥ +0.003 and is still awaiting its LB
verdict. Everything that came out null/negative is listed once in [§ Excluded](#excluded--tried-and-did-not-help-do-not-re-propose)
so it is not re-proposed. All scores **raw uncalibrated macro-F1** (project invariant — NO
logit calibration). `full_data CV` = held-out slice, high-variance, does **not** rank models —
**LB is the judge** (proven by the E8 CV→LB reversal).

## Leaderboard progression (single source of truth for "what the score is")

| Milestone | Model | LB | Δ vs prev | Lever that bought it |
|---|---|---:|---:|---|
| prior SOTA | `names_single` (calibrated) | 0.77427 | — | teammate baseline |
| E8 | qwen3_ls (richargs + LS + full_data) | 0.77921 | +0.0049 | **backbone+richargs+full_data+LS** |
| E8 | granite_ls (same recipe) | 0.77738 | — | (granite = fast #2, 5:06 vs qwen3 9:18) |
| E23 | granite_ls **+ TTA** lr1e-3 | 0.77931 | +0.00193 | **test-time adaptation** |
| E28 | granite **optuna t043fd** (single model) | **0.78155** | +0.0022 | **HPO** → single-model SOTA |
| E26 | granite **trio** (is3+aum06+e25c, uniform mean) | **0.78719** | +0.0056 | **ensembling** → CURRENT SOTA |

Net gain from the whole campaign so far: **+0.0129 LB** (0.77427 → 0.78719).

## Beneficial-lever board — the ingredients of the final model

| # | Lever | Exp | Effect | Evidence | In final recipe? |
|---|---|---|---|---|---|
| L1 | **richargs serialization** | E2/E21/E25 | +0.006–0.007 vs teammate `names`; = `richmeta` | E21 names 0.7731 < richargs 0.7803; E25c richmeta 0.7801 ≈ richargs | ✅ **yes** — the input format |
| L2 | **full_data folding** (train on all incl. most of val) | E8 | recipe foundation; enables the LB numbers above | champion recipe = `--full_data` | ✅ **yes** (LB judges, not the contaminated CV) |
| L3 | **Label smoothing ε=0.1** | E9 | **+0.0107 CV** — biggest single training lever | E9 loss screen: LS 0.7565 > CE 0.7458; wce +0.0083 #2 but < LS | ✅ **yes** — in every champion model |
| L4 | **Optuna-tuned HPs** (lr 2.6e-5 · eff_batch 8 · 5ep, best-epoch) | E28 | **single-model LB 0.78155** (+0.0042 vs granite_ls) | fold-0 +0.0138 vs champion HP anchor; top-8 all clear +0.009 | ✅ **yes** — the single-model core |
| L5 | **Ensembling — uniform softmax mean of diverse members** | E26 | **LB 0.78719** (+0.0079) — biggest jump; current SOTA | trio is3+aum06+e25c; uniform beats every fitted combiner (E29) & distillation (E30) | ✅ **yes** — the top layer; trio = direct-package ceiling (4th breaks caps) |
| L6 | **Test-time adaptation** (SHOT/IM entropy-min, 45 encoder LN affines) | E23 | **LB 0.77931** (+0.00193 vs granite_ls) | dose-response: lr1e-3 optimum; +2:40 inference | 🟡 **optional** — fits granite's 5:06 budget, NOT qwen3 (9:18); excluded from E26 (adapts each member → +2:40 ×N) |
| L7 | **AWP adversarial training** (weight perturbation, flat minima) | E34 | **+0.0071 CV** — best single-lever since LS | AWP 0.7804 vs anchor 0.7733; refutes E13 flatness prior | 🔲 **candidate** — LB probe + diverse ensemble member pending |
| L8 | **FGM adversarial** (embedding perturbation) | E32-A1 / E34-A | +0.0017 CV — marginal | 0.7750 vs anchor 0.7733; AWP beat it +0.0054 | 🔲 **member candidate only** — sub-gate as a single-model lever |

## Recommended final-model recipe

Two deliverables, both built from the levers above:

### A. Best single model (fast, 5:0x, fits any budget)
```
granite-311m
  · richargs serialization            (L1)
  · --full_data best-epoch            (L2)
  · --loss ls  (ε per tuned config)   (L3)
  · optuna HPs: lr 2.6e-5, eff_batch 8, ~5 epochs, best-epoch snapshot   (L4)
  · + AWP  (--awp_gamma 1e-3 --awp_lr 1e-4 --awp_start_epoch 1.0)        (L7, pending LB)
  · --init_seed 42                     (keeps it ensemble/soup-compatible)
```
Current best single = E28 **t043fd = LB 0.78155**. Adding AWP (L7, +0.0071 CV) is the next
single-model probe — if it holds on LB it becomes the new single-model SOTA and a member.

### B. Best overall — 3-member uniform ensemble (current SOTA, 6:52/836M)
Uniform softmax mean of 3 **diverse** granite members, packaged with shared vocab-prune + fp16
(the E26 packaging). The trio is the direct-package ceiling under the 1GB / 10-min caps — a 4th
member breaks both. Member selection: prefer **highest F1 × most diverse** (recipe /
serialization / seed disagreement is the currency uniform averaging converts).

- **Certified trio (shipping SOTA):** is3 (richargs) + aum06 (richargs) + e25c (richmeta) → **0.78719**.
- **Upgrade path:** swap in stronger/more-diverse members now available —
  E28 tuned granites (t043fd / t017fd / t048fd / t023fd, distinct ε/lr) and an **AWP** member
  (L7, different training dynamics ⇒ different errors). Rebuild the trio from the strongest
  diverse triplet; LB judges (the slice mis-ranks ensembles — E8/E26 lesson).

TTA (L6) can bolt onto a granite ensemble member but costs +2:40 **per member** → only viable
on a single granite, not the trio.

## Excluded — tried and did NOT help (do not re-propose)

Every one of these is a closed negative; kept here so the final push doesn't relitigate them.

| Lever | Exp | Verdict |
|---|---|---|
| First-step specialist (zero/strip-history) | E1 | −0.11 vs generalist — weakness is intrinsic |
| Two-model per-class-τ fallback | E6 | fails robustness (net-neg on 2/4 held-out slices) |
| TAPT continued pretraining | E11/E11b | +0.0134 **standalone** but **redundant** with LS+full_data (stack −0.0034) |
| Model soup (multi-seed) | E12 | cratered −0.056 (different-init members not mode-connected); deferred, needs shared-init |
| SAM fine-tune | E13 | closed — near-zero run variance = no flatness headroom (but AWP/L7 refuted this for *weight* perturbation) |
| `names` serialization (teammate format) | E21 | −0.0072 vs richargs — format is not his edge |
| Coreset / drop-noisy data selection | E22 | screen win but **null under LS champion** (LS absorbs the noise) |
| PGD adversarial | E34-B | −0.0060 — one-step FGM assumption dropped, but hurts |
| Upstream token/field selection | E24 | null for accuracy (efficiency only) |
| Teammate recipe repro | E25 | no edge; richmeta ≈ richargs; champion stands |
| Ensemble→single distillation | E26-1B / E30 | closed twice — trio edge is inference-time averaging, not distillable |
| Learned stacking / gating over members | E29 | uniform mean survives all 9 fitted combiners on honest 70k OOF |
| Error-prediction flip rule | E27 | LB-neutral (−0.00011); own-logit adjustment is information-bounded |
| NEFTune / hist_dropout / R-Drop(?) | E32-A3/B1 | flat, sub-gate (A3 0.7794, B1 0.7786); R-Drop A2 still running |
| Inference-time translation | E15a | NO-GO (93 min/30k ≫ 10-min budget) |

Secondary note — **wce** (E9, +0.0083) and **richmeta** (E25c, ≈ richargs) both clear noise but
are dominated by an included lever (LS / richargs); not promoted, available as diversity levers
for ensemble members.

## Cross-reference — efficiency levers that are accuracy-safe (see compression board)

These don't raise accuracy but are free/positive and let a compressed member fit a spare
ensemble/TTA slot. Full detail in [compression/results.md](../compression/results.md).

| Lever | Exp | Accuracy | Benefit |
|---|---|---|---|
| FFN low-rank factor + recovery (qwen3) | E4 | **+0.0045** (also a regularizer) | ~15% params ↓ |
| Depth-prune 28→14 + recovery (qwen3) | E16 | −0.0005 (free) | ~2× speed |
| grad-checkpoint auto-off (granite) | E19 | bit-neutral | 1.3–1.4× faster training |
| vocab-prune + fp16 | recipe | neutral | the size levers (every zip) |

## Open / pending — could still move the final number

| Item | Exp | Status | Decision gate |
|---|---|---|---|
| **AWP LB probe** | E34 | 🥇 +0.0071 CV, PROMOTE | single-model LB > 0.78155 ⇒ new SOTA + trio member |
| R-Drop | E32-A2 | 🏃 running (healthy GPU) | ≥ +0.003 vs anchor to gate |
| Model-level MoE (jointly trained) | E33 | 🏃 4 runs on vast | beat uniform-mean-of-own-experts null |
| Noise-robust losses (GCE/SCE/APL/ELR) | E35 | 🔲 queued | **beat LS**, not CE (E22 prior is negative) |
| Optuna round 2 | E28 | design | tighten around t043 basin |
| Trio member refresh (t043fd/AWP swap-in) | E26+E28+E34 | 🔲 | LB > 0.78719 |
</content>
</invoke>
