# Branch: performance-boost — performance program board (what actually improved the score)

Parallel to [compression/results.md](../compression/results.md): that board tracks **size↓ /
speed↑**; this one tracks **accuracy↑**. It is the curated shortlist — of the ~37 experiments
run, only the levers that **demonstrably raised macro-F1** (or are strong pending candidates)
are collected here, so the final-model recipe can be assembled from them without re-reading the
full [EXPERIMENTS.md](../EXPERIMENTS.md) dashboard.

**Selection rule for inclusion:** a lever earns a row only if it either (a) moved the **LB**
positively, or (b) moved the **from-scratch CV anchor** by ≥ +0.003 and is still awaiting its LB
verdict. Everything that came out null/negative is listed once in [§ Excluded](#excluded--tried-and-did-not-help-do-not-re-propose)
so it is not re-proposed. All scores **raw uncalibrated macro-F1** (project invariant — NO
logit calibration). `full_data CV` = held-out slice, high-variance, does **not** rank models —
**LB is the judge** (proven by the E8 CV→LB reversal).

_Last updated 2026-07-13 (SWA discarded; granite AWP+ELR resolved = +0.0009; qwen3 AWP/ELR transfer running; AWP wants ≥4 epochs)._

**Current focus = ONE model, not an ensemble.** The final single model stacks every beneficial
*single-model* lever: `richargs + full_data + LS + AWP + ELR + TTA`. Ensembling (L6) is set aside
(multi-model). ELR is the least-settled ingredient — marginal CV (+0.002 vs LS), within slice noise
on top of AWP. No LB yet — LB decides whether it stays. SWA/EMA was tested this session and
**discarded** (see Excluded). **Raw trained numbers for the E43/E44 runs live in the separate
[results_e43_e44.md](results_e43_e44.md), not here — this board is proven-useful levers only.**

## Leaderboard progression (single source of truth for "what the score is")

| Milestone | Model | LB | Δ vs prev | Lever that bought it |
|---|---|---:|---:|---|
| prior SOTA | `names_single` (calibrated) | 0.77427 | — | teammate baseline |
| E8 | qwen3_ls (richargs + LS + full_data) | 0.77921 | +0.0049 | **backbone+richargs+full_data+LS** |
| E8 | granite_ls (same recipe) | 0.77738 | — | (granite = fast #2, 5:10 vs qwen3 9:18) |
| E23 | granite_ls **+ TTA** lr1e-3 | 0.77931 | +0.00193 | **test-time adaptation** |
| E28 | granite **optuna t043fd** (single) | 0.78155 | +0.0022 | **HPO** |
| E34 | granite **+ AWP** (single) | **0.78557** | +0.0040 | **AWP adversarial** → single-model SOTA |
| E26 | granite **trio** is3+aum06+e25c (uniform mean) | 0.78719 | +0.0016 | **ensembling** |
| E28 | granite **optuna trio** t019+t023+t043 | **0.78780** | +0.00061 | **HPO × ensembling** → CURRENT SOTA |

Net gain from the whole campaign so far: **+0.0135 LB** (0.77427 → 0.78780).

## Beneficial-lever board — the ingredients of the final model

| # | Lever | Exp | Effect | Evidence | In final recipe? |
|---|---|---|---|---|---|
| L1 | **richargs serialization** | E2/E21/E25 | +0.006–0.007 vs teammate `names`; = `richmeta` | E21 names 0.7731 < richargs 0.7803; E25c richmeta 0.7801 ≈ richargs | ✅ **yes** — the input format |
| L2 | **full_data folding** (train on all incl. most of val) | E8 | recipe foundation; enables the LB numbers above | champion recipe = `--full_data` best-epoch (NOT `--all_data` — last-epoch tail trap, LB 0.75934) | ✅ **yes** (LB judges, not the contaminated CV) |
| L3 | **Label smoothing ε=0.1** | E9 | **+0.0107 CV** — biggest single training lever | E9 loss screen: LS 0.7565 > CE 0.7458; wce +0.0083 #2 but < LS | ✅ **yes** — in every champion model |
| L4 | **Optuna-tuned HPs** (lr 2.6e-5 · eff_batch 8 · ~5ep, best-epoch) | E28 | **single LB 0.78155**; **trio LB 0.78780** | fold-0 +0.0138 vs champion HP anchor; top-8 all clear +0.009 | ✅ **yes** — the single-model + trio core |
| L5 | **AWP adversarial training** (weight perturbation, flat minima) | E34 | **single LB 0.78557** (+0.0040 vs t043fd) — single-model SOTA, ties 2-granite pairs at 5:10 | AWP 0.7804 vs anchor 0.7733 (+0.0071 CV); refutes E13 flatness prior; slice under-predicted (+0.005 on LB) | ✅ **yes** — best single model; prime ensemble member |
| L6 | **Ensembling — uniform softmax mean of diverse members** | E26/E28 | **trio LB 0.78780** (+0.006 over best single) — current SOTA | uniform beats every fitted combiner (E29) & distillation (E30); fold screen predicted +0.0055 exactly | ✅ **yes** — the top layer |
| L7 | **Test-time adaptation** (SHOT/IM entropy-min, 45 encoder LN affines) | E23 | LB 0.77931 (+0.00193 vs granite_ls) | dose-response: lr1e-3 optimum; +2:40 inference | ✅ **yes** (single-model) — inference-time, fits granite's ~5:10 budget (NOT qwen3 9:18) |
| L8 | **ELR** (Early-Learning Regularization: `LS-CE + λ·log(1−⟨p,t⟩)`, λ=3 β=0.7) | E35-nr | **+0.0021** vs LS (56k/14k) — partially refutes E22 "LS absorbs all noise" | on top of AWP it's **within slice noise** (numbers → [results_e43_e44.md](results_e43_e44.md)) | 🟡 **marginal, LB-pending** — sub +0.003 gate; keep only if LB confirms. LB decides |

## Recommended final-model recipe

Two deliverables, both built from the levers above:

### A. Best single model — the full stack (THIS is the target deliverable)
```
granite-311m
  · richargs serialization            (L1)
  · --full_data best-epoch            (L2)
  · --loss ls  ε=0.1                  (L3)
  · + AWP  (--awp_gamma 1e-3 --awp_lr 1e-4 --awp_start_epoch 1.0)   (L5)
  · + ELR  (--elr_lambda 3 --elr_beta 0.7, LS-CE primary)          (L8, marginal/LB-pending)
  · + TTA at inference (SHOT/IM lr1e-3, 45 encoder LN affines)      (L7)
  · --init_seed 42
```
**Confirmed floor: E34 AWP alone → LB 0.78557** (single-model SOTA). The stack above adds ELR
(train-time, +0.002 CV) and TTA (inference-time, +0.002 LB on granite_ls) on top — **both
LB-unconfirmed on the combined model.** Order of certainty: AWP (LB ✅) > TTA (LB ✅ standalone)
> ELR (CV-only, and may be redundant under AWP — see L8). E35 (AWP-knob Optuna, running) can
still re-tune the recipe+AWP jointly and shift the optimum.

**AWP epochs:** use **≥4** (AWP is still climbing at ep3, not plateaued). Curve + numbers →
[results_e43_e44.md](results_e43_e44.md).

**qwen3 backbone transfer (E44, ✅ done):** AWP/ELR applied to qwen3 (E8b+LS backbone, LB-stronger).
**Transfer succeeds — qwen3 overtakes granite: LS+AWP 0.7847 > granite AWP 0.7822 (+0.0025)** (tied at
ep3, qwen3 out-climbs on ep4). AWP ≫ ELR holds; qwen3 needs the 4th epoch (still climbing → likely
under its ceiling). Slice under-ranks qwen3 → LB judges; catch = qwen3 ~9:18 needs compression to ship.
Full trained results (qwen3 2×2 + granite comparison) → [results_e43_e44.md](results_e43_e44.md).

**What to train / probe next (single-model, priority order):**
1. ✅ **Answered (granite): ELR is within-noise on top of AWP** (numbers → E43/E44 doc) — optional, keep only if LB confirms.
2. **Submit LS+AWP+ELR full_data** → first LB datum on ELR's real contribution.
3. **TTA-wrap the best of {AWP, AWP+ELR}** → submit → does TTA still add on the adversarial model?
4. **E35 AWP-Optuna winner** (`--full_data`) → the jointly-tuned recipe+AWP, should exceed 0.78557.
5. ❌ **SWA/EMA — closed** (E43): no raw gain on the AWP model.

### B. Best overall — 3-member uniform ensemble — LB 0.78780 (CURRENT SOTA, 6:39/842M)
Uniform softmax mean of 3 **diverse** granite members, shared vocab-prune + fp16 (E26/E28
packaging). Member selection: **highest F1 × most diverse** (recipe / serialization / seed / lr
disagreement is the currency uniform averaging converts).

- **Reigning SOTA:** E28 optuna trio **t019 + t023 + t043** → **0.78780** (diversity = lr-spread only).
- **Upgrade path (built, LB pending):** a **quad** now packages under the caps via aggressive
  vocab-prune (union 16,473 tokens → 247 MB/member fp16 → 917 MB total) — **the "4th member breaks
  caps" limit is solved.** `submit_0713_quad.zip` = **AWP** + t019 + e25c_richmeta + anchor
  (adds AWP's distinct regime + a serialization-diverse member). LB judges (slice mis-ranks
  ensembles — E8/E26 lesson). Also built, LB pending: base/selA/selC member-selection trios.

TTA (L7) can bolt onto a single granite member but costs +2:40 **per member** → viable on one
granite, not the ensemble.

## Excluded — tried and did NOT help (do not re-propose)

Every one of these is a closed negative; kept here so the final push doesn't relitigate them.

| Lever | Exp | Verdict |
|---|---|---|
| First-step specialist (zero/strip-history) | E1 | −0.11 vs generalist — weakness is intrinsic |
| Two-model per-class-τ fallback | E6 | fails robustness (net-neg on 2/4 held-out slices) |
| TAPT continued pretraining | E11/E11b | +0.0134 **standalone** but **redundant** with LS+full_data (stack −0.0034) |
| Model soup (multi-seed) | E12 | cratered −0.056 (different-init members not mode-connected); needs shared-init |
| SAM fine-tune | E13 | closed — no flatness headroom (but AWP/L5 refuted this for *weight* perturbation) |
| `names` serialization (teammate format) | E21 | −0.0072 vs richargs — format is not his edge |
| Coreset / drop-noisy data selection | E22 | screen win but **null under LS champion** (LS absorbs the noise) |
| Upstream token/field selection | E24 | null for accuracy (efficiency only) |
| Teammate recipe repro | E25 | no edge; richmeta ≈ richargs; champion stands |
| Ensemble→single distillation | E26-1B / E30 | closed twice — trio edge is inference-time averaging, not distillable |
| Learned stacking / gating over members | E29 | uniform mean survives all 9 fitted combiners on honest 70k OOF |
| Error-prediction flip rule | E27 | LB-neutral (−0.00011); own-logit adjustment is information-bounded |
| **FGM / PGD embedding perturbation** | E32-A1 / E34-A,B | **closed negative**: FGM 0.7712–0.7750 (marginal-to-hurt), PGD −0.0060. Only **weight** perturbation (AWP/L5) won — embedding perturbation lost |
| **R-Drop / NEFTune / hist_dropout** (training-time noise) | E32-A2/A3/B1 | all flat-to-negative (R-Drop 0.7716, NEFTune 0.7794, hist_dropout 0.7786); no arm beat champion 0.7803 |
| **Model-level MoE** (jointly trained experts + router) | E33 | clean negative −0.010…−0.012 (0.7686–0.7702); shared-trunk experts weak, 1–2 die/run; gate wins only by pruning dead experts |
| **DropHead** (structured attn-head dropout) | E47/E49 | ⚠️ screen-win but **LB-negative**: won 14k screen +0.0044 (0.7826 vs 0.7782, 56k/6ep) yet **LB 0.78866 < AWP t031 0.79300** (SUBMISSIONS row 27). Screen delta was within DropHead's ~0.005 stochastic-mask run-variance (single-seed); 3.5k slice predicted the loss (0.7839<0.7856). E49 variants (schedule/granularity/correlated/ramp/LayerDrop/DropFFN) all 8ep-depressed + single-seed; best = layer-ramp 0.7824. ⚠️ **NOT fully closed** — a *controlled* Optuna (pinned epochs, multi-seed) could still resolve whether the +0.0044 is real (see Open) |
| **Grouped/negative label smoothing · Child-Tuning-D** | E47 | 14k-screen null-to-neg: grouped ±LS −0.0015, neg-LS +0.0001, child-tuning +0.0001 (LS ε=0.1365 already ~optimal under AWP) |
| **Inference-time multi-view averaging** | E36-A | flat (best +0.0003, noise); diversity is model-level not view-level |
| Sub-sequence / prefix expansion | E36 | near no-op — organizers already per-step-expanded (70k→73k = +4.6%) |
| Inference-time translation | E15a | NO-GO (93 min/30k ≫ 10-min budget) |
| **Weight EMA / SWA** (checkpoint averaging on the AWP model) | **E43** | **discarded** — no raw gain on the AWP model (calibration-only); too few epochs to have a tail + AWP already flat-minima → redundant. Numbers + the `swa_average.py` serialization-bug note → [results_e43_e44.md](results_e43_e44.md) |

Secondary note — **wce** (E9, +0.0083) and **richmeta** (E25c, ≈ richargs) both clear noise but
are dominated by an included lever (LS / richargs); not promoted, kept as diversity levers for
ensemble members (richmeta is the serialization-diverse member in the quad).

## Cross-reference — efficiency levers that are accuracy-safe (see compression board)

These don't raise accuracy but are free/positive and let a compressed member fit a spare
ensemble/TTA slot. Full detail in [compression/results.md](../compression/results.md).

| Lever | Exp | Accuracy | Benefit |
|---|---|---|---|
| FFN low-rank factor + recovery (qwen3) | E4 | **+0.0045** (also a regularizer) | ~15% params ↓ |
| Depth-prune 28→14 + recovery (qwen3) | E16 | −0.0005 (free) | ~2× speed |
| grad-checkpoint auto-off (granite) | E19 | bit-neutral | 1.3–1.4× faster training |
| vocab-prune + fp16 | recipe | neutral | the size levers — aggressive union-prune is what makes the quad fit |

## Open / pending — could still move the final number

**Single-model track (current focus):**

| Item | Exp | Status | Decision gate |
|---|---|---|---|
| LS+AWP vs LS+AWP+ELR (granite) | E35-nr | ✅ done: ELR within-noise on AWP (→ E43/E44 doc) | ELR marginal on AWP → keep only if LB confirms |
| **qwen3 AWP / ELR / AWP+ELR** transfer | **E44** | ✅ done: **qwen3 LS+AWP 0.7847 > granite AWP 0.7822** (+0.0025) → E43/E44 doc | AWP transfers + qwen3 overtakes; LB judges (qwen3 ~9:18 → needs compression to ship) |
| **LS+AWP+ELR LB** | E34+E35-nr | 🔲 build+submit after runs finish | LB vs AWP-alone 0.78557 |
| **AWP(+ELR) + TTA** stack | E23+E34 | 🔲 not yet run | TTA-wrap → does test-time adapt still add on adversarial model? |
| **AWP-knob Optuna winner** (joint AWP+recipe) | E35-awp | 🏃 running (16 workers, 8×5090) | top-2–3 `--full_data` → LB > 0.78557 |
| Sequence pooling (mean / attn pool) | E37 | 🔲 queued (design only) | ≥ +0.003 vs cls anchor |
| **DropHead controlled Optuna** | E47/E49 | 🔲 staging on new 8×5090 (44900306) | joint (p·schedule·granularity·layer_ramp + recipe), **epochs PINNED at 4** (E38 lesson), **multi-seed** to see past ~0.005 DropHead noise; gate: seed-mean 14k > AWP anchor by > noise → `--full_data` → LB > 0.79300. Motivated by clean 14k DropHead>AWP +0.0044 + E49 layer-ramp 0.7824 |
| ~~Weight EMA / SWA~~ | **E43** | ❌ closed (no raw gain; calibration-only) | — see Excluded |

**Ensemble track (set aside per current single-model focus, but built & LB-pending):**

| Item | Exp | Status |
|---|---|---|
| Quad LB (AWP + t019 + richmeta + anchor) | E26/E28/E34 | ✅ built (917M), LB pending — 4th-member cap solved |
| Member-selection trios (base / selA / selC) | E26 | ✅ built, LB pending |
</content>
