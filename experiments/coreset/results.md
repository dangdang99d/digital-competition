# Branch: coreset — drop-noisy data selection to raise macro-F1

Logical group `research/coreset` (committed on `research/token-selection`). Objective: raise
**macro-F1** by training on a cleaned subset — the *drop-noisy / data-centric denoising* family
(remove mislabeled samples), **not** keep-hard coreset, and **not** compute reduction.
Screen backbone: **granite-311m** (fast; E9/E11 precedent). All scores **raw uncalibrated
macro-F1** (project invariant — NO calibration).

Baselines: historical E9 CE control **0.7458** · in-pipeline `coreset_base` (identical harness,
full 56k) **0.7498** · deployment champion E8a+LS richargs full_data **0.7803** (3.5k held-out).

## Summary

| Step | What | Status | Result | Verdict |
|---|---|:--:|---|---|
| §1 | suitability probe (removable noise vs ambiguity) | ✅ | ~6% two-model-consensus mislabels; 40% of errors low-conf; errors pile on synonymous file-ops | 🟡 CONDITIONAL GO — surgical only; blanket drop-wrong predicted to hurt |
| §2 | screen: 6 scorers → 10 keep-set retrains (granite v1+CE, standard split) | ✅ | **pvi06 +0.0063 · aum06 +0.0048**; cleanlab/forget ~flat; **cart/el2n hurt, worse when aggressive** | 🟢 surgical denoise CONFIRMED on the screen recipe |
| §3 | gate analysis — can the denoised model rule out hard at inference? | ✅ | AUROC(MSP→hard) 0.830→0.835; win is easy-tier F1 +0.010 | gate **inherited, not improved** — no routing lever |
| §4 | champion confirm — pvi06/aum06 keep-sets × E8a+LS richargs full_data | ✅ | **pvi06 0.7700 (−0.0103) · aum06 0.7766 (−0.0037)** vs champion 0.7803 | ❌ **does NOT transfer to the champion recipe** (3.5k held-out; LB untested) |
| — | Group 3 loops (co-teaching / MentorNet / DivideMix / SELFIE) | ⛔ | never run | gated on a big denoise gain — not met |

## §1 · Suitability probe (no training; qwen3 champion val-OOF + hist0 as 2nd model)

- **What:** how much of our 23.5% val error is *removable noise* (mislabels) vs *irreducible
  ambiguity*? Drop-noisy only helps for the former.
- **Result** ([figure](figures/suitability_error_profile.png)): **6.26%** of val = two independent
  architectures confidently agree on a non-label class (mislabel signature) · **40.5% of errors**
  are low-confidence (MSP<0.5) near-boundary ambiguity · errors concentrate on the synonymous
  file-ops (`list_directory` 48% · `read_file` 46% · `glob_pattern` 43% · `grep_search` 32%).
- **Verdict:** a real ~6% removable population exists → try *surgical* removal only; blanket
  "drop everything predicted wrong" would starve rare/confusable classes → predicted macro-F1 loss.

## Decisions (methodology, fixed before the screen)

- **Score & retrain with FRESH granite from HF base** — never a fine-tuned checkpoint: a model
  that already fit the noise can't un-see it, and its in-sample train predictions are invalid for
  confident learning. Group-1 OOF = k-fold fresh granite; Group-2 dynamics = one instrumented
  baseline run (`finetune.py --log_dynamics`). Retrains use `--keep_indices` (identical recipe,
  only the train rows differ).
- **Eval:** one raw forward pass over the 14k val → overall macro-F1 (decision metric) + per-class
  F1 (rare-class guard) + tier macro-F1 via boolean masks. **Val hardness ruler = qwen3 champion
  cached OOF logits** (`analysis/cache/qwen3_val_logits.npz`): easy = qwen3-correct ·
  suspect-mislabel = qwen3 confident-wrong (MSP>0.6) · ambiguous = qwen3 low-conf wrong. Never
  partition by the evaluated model's own MSP (circular). Tiers: easy 10714 · ambig 1900 ·
  suspect 1386 · cleanlab-clean 11744.
- **Two "hard" notions:** suspect-mislabel → tier F1 is *confounded* (a better model scores worse
  against a wrong label); ambiguous → F1 meaningful. The "did dropping help" read = easy/clean-val.
- Weights note: champion weights live in `submissions/*.zip`, never `output/` (see memory
  `qwen3-champion-weights-loading`); this experiment needs none — everything trains from HF base.

## §2 · Screen — 6 scorers, 10 keep-set retrains

- **Model:** granite-311m, v1@512, CE, standard split (56k train / 14k val), seed 42.
- **Baseline:** in-pipeline `coreset_base` = **0.7498**.
- **Change:** drop the scorer's noisiest 6%/15% of train (`--keep_indices`), retrain identically.
  Scorers: **C1 cleanlab** (4-fold OOF self-confidence) · **C2 AUM** (mean margin) · **C4
  cartography drop-hard** (lo-conf ∧ lo-var; protect hi-var) · **C5 forgetting** (never-learned) ·
  **C6 EL2N** (early-epoch error norm) · **C7 PVI** (OOF gold-prob vs class prior, bits).
- **Result** (raw val macro-F1; Δ vs 0.7498; full per-tier/per-class tables in
  [eval_results.md](eval_results.md); where each method cuts:
  [datamap_drops](figures/datamap_drops.png) · [datamap_pvi](figures/datamap_pvi.png) ·
  [datamap](figures/datamap.png)):

  | keep-set | drop | val F1 | Δ | verdict |
  |---|---|---|---|---|
  | **pvi06** | 6% | **0.7561** | **+0.0063** | 🥇 |
  | **aum06** | 6% | **0.7546** | **+0.0048** | 🥈 |
  | aum15 | 15% | 0.7542 | +0.0044 | robust to drop level |
  | pvi15 | 15% | 0.7514 | +0.0016 | still positive |
  | forget | 10.1% | 0.7505 | +0.0007 | flat |
  | cl | 20.2% | 0.7500 | +0.0002 | flat (over-drops) |
  | el2n06 / el2n15 | 6 / 15% | 0.7458 / 0.7368 | −0.0040 / −0.0130 | ❌ eats the ambiguous wing |
  | cart06 / cart15 | 6 / 15% | 0.7435 / 0.7419 | −0.0063 / −0.0079 | ❌ quadrant cut misfires |

- **Verdict:** exactly the §1 prediction — **surgical mislabel removal helps (+0.005–0.006),
  blanket drop-hard hurts**. PVI/AUM (continuous mislabel rankings) win; PVI's edge over AUM =
  OOF scoring also catches *memorized* noise invisible to training dynamics (visible as
  high-confidence red points in [datamap_drops](figures/datamap_drops.png)). Tier read: on the
  label-trustworthy subset the win is bigger — clean-val +0.011 (pvi06 0.8762 vs 0.8656), easy
  +0.009; hard-tier F1 drops as expected (mislabel confound). Per-class: the confusable file-ops
  hold or improve (pvi06: `list_directory` +0.037, `read_file` +0.040) — no rare-class damage.

## §3 · Gate analysis — inference-time hard-sample rejection (user ask 2026-07-09)

- **What:** does denoise training make the model's own confidence a better gate (confident on
  easy, unconfident on hard)? `gate_analysis.py`, raw fp32 logits, qwen3-ruler tiers, logits
  cached `analysis/cache/coreset_gate_logits.npz`. Figures:
  [msp_tiers](figures/gate_msp_tiers.png) · [risk_coverage](figures/gate_risk_coverage.png) ·
  [suspect_behavior](figures/gate_suspect_behavior.png).
- **Result:**

  | model | overall | easy F1 | ambig F1 | suspect F1 | MSP e/a/s | AUROC hard / own-err | gate@76.5% cov (oracle) |
  |---|---|---|---|---|---|---|---|
  | base | 0.7500 | 0.9107 | 0.4239 | 0.1767 | 0.88/0.53/0.77 | 0.830 / 0.845 | 0.8054 (0.9107) |
  | pvi06 | 0.7565 | **0.9204** | 0.3929 | 0.1577 | 0.89/0.52/0.79 | 0.835 / 0.836 | 0.8050 (0.9204) |
  | aum06 | 0.7545 | **0.9209** | 0.3935 | 0.1599 | 0.90/0.52/0.80 | 0.833 / 0.841 | 0.8024 (0.9209) |
  | cl | 0.7501 | 0.9178 | 0.4112 | 0.1529 | 0.98/0.91/0.97 ⚠️ | 0.767 / 0.787 | inflated by MSP ties — ignore |
  | cart15 | 0.7430 | 0.8960 | 0.4100 | 0.1457 | 0.86/0.56/0.74 | 0.817 / 0.835 | 0.7805 (0.8960) |

- **Verdict:** **the gate is inherited, not improved** — AUROC flat at the E20 model-independent
  ~0.85 ceiling; the denoise win lives in easy-tier F1 (+0.010). Hard splits into two populations:
  *ambiguous* rows are already flagged by low MSP (~0.52); *mislabeled* rows can't be flagged —
  every model (incl. base) confidently predicts the qwen3-consensus class on 81–85% of them (only
  9–12% the label), independently confirming the mislabel reading. MSP-gating at 76.5% coverage
  reaches 0.805 vs the 0.92 oracle → no routing lever (E6/E20 again). ⚠️ `cl` trains to uniform
  overconfidence (MSP ≈ 0.97 on every tier) → worst detector; a warning for aggressive cleaning.

## §4 · Champion confirm — does the gain transfer to the deployment recipe? ❌

- **Model/recipe:** granite E8a+LS **richargs · full_data · LS ε0.1 · 3 ep · lr 2e-5 · seed 42**
  (exact champion recipe; same 3.5k held-out) — from HF base, only the train rows filtered.
- **Keep-set plumbing:** `--keep_indices` composes with `--full_data`, but the keep npy covers
  only the original 56k train — the 10.5k val-derived rows added by full_data would be silently
  dropped. Fix (no code change): extended files `keepsets/{pvi,aum}_drop06_fulldata.npy` =
  keep ∪ va → verified `train 66500 → 63140 (dropped 3360 noisy)`.
- **Result** (raw `val_macro_f1`, `output/pat/ft_results_e22_champion.csv`, runs 2026-07-09):

  | run | val F1 | Δ vs champion 0.7803 |
  |---|---|---|
  | e22_pvi06_champion | 0.7700 | **−0.0103** |
  | e22_aum06_champion | 0.7766 | **−0.0037** |

- **Verdict:** ❌ **the screen gain does NOT transfer** — both land *below* the champion on the
  shared held-out (pvi06 well beyond the ±0.003 slice noise). Candidate explanations, untested:
  (a) **LS already absorbs label noise** (soft targets damp mislabel gradients → denoising is
  redundant, and dropping 3360 rows just loses data); (b) keep-sets were scored under **v1+CE on
  the standard split** — importance may not transfer to richargs+LS+full_data; (c) 3.5k-slice CV
  has mis-ranked LB before (E8) — but −0.010 is large, so an LB submit on these weights is hard to
  justify. If the line is revisited: rescore (PVI k-fold OOF) *under the champion recipe itself*,
  or test drop-vs-LS interaction (pvi06 × CE full_data).

## Assets & reproduce

- Scripts (`experiments/coreset/`): `confident_learning.py` (C1, k-fold OOF, `--folds/--merge`
  for multi-GPU) · `score_dynamics.py` (C2/C4/C5/C6 from the dynamics npz) · `pvi.py` (C7, reuses
  C1's OOF) · `coreset_eval.py` (per-tier/per-class eval → `eval_results.md`) · `gate_analysis.py`
  · `plot_drop_maps.py` · orchestrators `run_coreset*.py` (GPU-slot queue over the DAG).
- Plumbing (`src/finetune.py`): `--keep_indices <npy>` (absolute-index train filter) ·
  `--log_dynamics <npz>` (per-epoch train softmax callback).
- Caches: `analysis/cache/coreset_dyn_granite.npz` (dynamics E×N×C) · `coreset_oof_granite.npz`
  (4-fold OOF) · `coreset_gate_logits.npz` (gate-analysis val logits) · keep-sets in `keepsets/`
  (npy, gitignored — regenerate via the scorers). Logs: `sbatch/logs/coreset_*`, `e22_*_champion.log`.
