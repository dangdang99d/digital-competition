# Branch: coreset — drop-noisy data selection to raise macro-F1

Logical group `research/coreset` (committed on `research/token-selection`). Objective: raise
**macro-F1** by training on a cleaned subset — the *drop-noisy / data-centric denoising* family
(remove mislabeled samples), **not** keep-hard coreset, and **not** compute reduction.
Screen backbone: **granite-311m** (fast; E9/E11 precedent). All scores **raw uncalibrated
macro-F1** (project invariant — NO calibration).

**MSP** (max softmax probability, used throughout) = `max_c softmax(logits)_c` — the model's
confidence in its own top prediction (one number per row, label-free, computable at test time).
Low MSP = unsure. Used as the confidence gate/ruler signal; always from raw fp32 logits.

Baselines: historical E9 CE control **0.7458** · in-pipeline `coreset_base` (identical harness,
full 56k) **0.7498** · deployment champion E8a+LS richargs full_data **0.7803** (3.5k held-out).

## Summary

| Step | What | Status | Result | Verdict |
|---|---|:--:|---|---|
| §1 | suitability probe (removable noise vs ambiguity) | ✅ | ~6% two-model-consensus mislabels; 40% of errors low-conf; errors pile on synonymous file-ops | 🟡 CONDITIONAL GO — surgical only; blanket drop-wrong predicted to hurt |
| §2 | screen: 6 scorers → 10 keep-set retrains (granite v1+CE, standard split) | ✅ | **pvi06 +0.0063 · aum06 +0.0048**; cleanlab/forget ~flat; **cart/el2n hurt, worse when aggressive** | 🟢 surgical denoise CONFIRMED on the screen recipe |
| §3 | gate analysis — can the denoised model rule out hard at inference? | ✅ | AUROC(MSP→hard) 0.830→0.835; win is easy-tier F1 +0.010 | gate **inherited, not improved** — no routing lever |
| §4 | champion confirm — pvi06/aum06 keep-sets × E8a+LS richargs full_data | ✅ | **pvi06 0.7700 (−0.0103) · aum06 0.7766 (−0.0037)** vs champion 0.7803 | ❌ **does NOT transfer to the champion recipe** (3.5k held-out; LB untested) |
| §5 | group-level confusion — TP/FP/TN/FN on the 4 action groups | ✅ | group acc **0.989–0.995** (class acc ~0.75); containment P(true∈pred group \| class wrong) **0.96–0.98** | error is almost entirely **intra-group** — group prediction is near-solved, lever = intra-group disambiguation |
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
- **Val tier definitions** (fixed by the qwen3 ruler; the tiers partition by the RULER's
  correctness/confidence — tier F1 in the tables below is then computed for the *evaluated*
  (granite) models on those rows, so hard-tier F1 is nonzero even though the ruler got them wrong):

  | tier | definition (qwen3 ruler on 14k val) | n | meaning | tier-F1 trustworthy? |
  |---|---|---|---|---|
  | easy | qwen3 predicted the label correctly | 10714 | label consistent with text; model-handleable | ✅ yes |
  | ambig | qwen3 wrong, MSP ≤ 0.6 (unsure) | 1900 | text genuinely unclear between classes | ✅ yes |
  | suspect | qwen3 wrong, MSP > 0.6 (confident) | 1386 | label itself likely wrong (mislabel signature) | ⚠️ no — grades against a bad label; a better model scores *worse* |
  | clean-val | val minus cleanlab-flagged rows | 11744 | label-trustworthy subset | ✅ yes — the decision metric |
- Weights note: champion weights live in `submissions/*.zip`, never `output/` (see memory
  `qwen3-champion-weights-loading`); this experiment needs none — everything trains from HF base.

## §2 · Screen — 6 scorers, 10 keep-set retrains

- **Model:** granite-311m, v1@512, CE, standard split (56k train / 14k val), seed 42.
- **Baseline:** in-pipeline `coreset_base` = **0.7498**.
- **Change:** retrain identically on a filtered train set (`--keep_indices`); 10 keep-sets =
  6 scorers × the drop rules below (no C3 — number never assigned). Two signal sources:
  **4-fold OOF** (train on 3 folds, score the held-out fold — out-of-sample, so it also catches
  noise the model *memorized*) and **training dynamics** (per-epoch train softmax from one
  instrumented baseline run, `--log_dynamics`).

  | scorer → keep-set(s) | signal | score per train row | drop rule |
  |---|---|---|---|
  | C1 confident learning → `cl` | OOF | cleanlab `find_label_issues` on (labels, OOF probs) — flags rows whose given label is inconsistent with the confident joint (Northcutt'21) | drop every flagged row (20.2%) |
  | C2 AUM → `aum06/15` | dynamics | margin = mean over epochs of p_gold − max other-class prob (Pleiss'20); low/negative = the label fights the gradient signal all training long | drop lowest 6% / 15% |
  | C4 cartography → `cart06/15` | dynamics | confidence = mean_e p_gold · variability = std_e p_gold (Swayamdipta'20); hardness = (1−conf)·1[var<median] — cuts the hard-to-learn quadrant, protects hi-var ambiguous | drop highest-hardness 6% / 15% |
  | C5 forgetting → `forget` | dynamics | never-learned = predicted correctly in 0 epochs (Toneva'19) | drop all never-learned (10.1%) |
  | C6 EL2N → `el2n06/15` | dynamics | ‖softmax − onehot‖₂ at the first epoch (Paul'21) — error norm before memorization sets in | drop highest 6% / 15% |
  | C7 PVI → `pvi06/15` | OOF (reuses C1's) | log₂ p_OOF(gold\|x) − log₂ prior(gold), bits (Ethayarajh'22); negative = the input makes the gold label *less* likely than prior-guessing ≈ unlearnable/mislabeled | drop lowest (most negative) 6% / 15% |
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

## §5 · Group-level confusion — TP/FP/TN/FN on the 4 action groups (user ask 2026-07-09)

- **What:** the 14 classes form 4 confusion groups (`src/data.ACTION_GROUPS`: **explore** =
  read_file/grep_search/list_directory/glob_pattern · **edit** = edit_file/write_file/apply_patch ·
  **execute** = run_bash/run_tests/lint_or_typecheck · **noncode** = ask_user/plan_task/
  web_search/respond_only). Hypothesis: even when the class prediction is wrong, the true label
  sits in the *predicted group* — so the residual error is intra-group disambiguation.
  `group_confusion.py` on the cached §3 val logits + qwen3 ruler (14k val, raw, NO calibration).
- **Result — class vs group accuracy** (containment = P(pred group == true group | class wrong)):

  | model | class acc | group acc | class-err rate | containment |
  |---|---|---|---|---|
  | qwen3 (champion ruler) | 0.7653 | 0.9931 | 0.2347 | 0.9705 |
  | base | 0.7545 | **0.9946** | 0.2455 | **0.9779** |
  | pvi06 | 0.7581 | 0.9913 | 0.2419 | 0.9640 |
  | aum06 | 0.7578 | 0.9920 | 0.2422 | 0.9670 |
  | cl | 0.7566 | 0.9897 | 0.2434 | 0.9577 |
  | cart15 | 0.7492 | 0.9928 | 0.2508 | 0.9712 |

- **Group-level one-vs-rest TP/FP/TN/FN** (predicted group vs true group; group sizes:
  explore 5756 · edit 3495 · execute 2383 · noncode 2366):

  | model | group | TP | FP | TN | FN | precision | recall | ovr acc |
  |---|---|---|---|---|---|---|---|---|
  | qwen3 (ruler) | explore | 5719 | 37 | 8207 | 37 | 0.9936 | 0.9936 | 0.9947 |
  | qwen3 (ruler) | edit | 3483 | 22 | 10483 | 12 | 0.9937 | 0.9966 | 0.9976 |
  | qwen3 (ruler) | execute | 2356 | 24 | 11593 | 27 | 0.9899 | 0.9887 | 0.9964 |
  | qwen3 (ruler) | noncode | 2345 | 14 | 11620 | 21 | 0.9941 | 0.9911 | 0.9975 |
  | base | explore | 5729 | 28 | 8216 | 27 | 0.9951 | 0.9953 | 0.9961 |
  | base | edit | 3483 | 15 | 10490 | 12 | 0.9957 | 0.9966 | 0.9981 |
  | base | execute | 2360 | 11 | 11606 | 23 | 0.9954 | 0.9903 | 0.9976 |
  | base | noncode | 2352 | 22 | 11612 | 14 | 0.9907 | 0.9941 | 0.9974 |
  | pvi06 | explore | 5699 | 37 | 8207 | 57 | 0.9935 | 0.9901 | 0.9933 |
  | pvi06 | edit | 3483 | 23 | 10482 | 12 | 0.9934 | 0.9966 | 0.9975 |
  | pvi06 | execute | 2357 | 44 | 11573 | 26 | 0.9817 | 0.9891 | 0.9950 |
  | pvi06 | noncode | 2339 | 18 | 11616 | 27 | 0.9924 | 0.9886 | 0.9968 |
  | aum06 | explore | 5700 | 26 | 8218 | 56 | 0.9955 | 0.9903 | 0.9941 |
  | aum06 | edit | 3483 | 23 | 10482 | 12 | 0.9934 | 0.9966 | 0.9975 |
  | aum06 | execute | 2356 | 35 | 11582 | 27 | 0.9854 | 0.9887 | 0.9956 |
  | aum06 | noncode | 2349 | 28 | 11606 | 17 | 0.9882 | 0.9928 | 0.9968 |
  | cl | explore | 5703 | 62 | 8182 | 53 | 0.9892 | 0.9908 | 0.9918 |
  | cl | edit | 3484 | 28 | 10477 | 11 | 0.9920 | 0.9969 | 0.9972 |
  | cl | execute | 2345 | 32 | 11585 | 38 | 0.9865 | 0.9841 | 0.9950 |
  | cl | noncode | 2324 | 22 | 11612 | 42 | 0.9906 | 0.9822 | 0.9954 |
  | cart15 | explore | 5726 | 51 | 8193 | 30 | 0.9912 | 0.9948 | 0.9942 |
  | cart15 | edit | 3486 | 14 | 10491 | 9 | 0.9960 | 0.9974 | 0.9984 |
  | cart15 | execute | 2341 | 17 | 11600 | 42 | 0.9928 | 0.9824 | 0.9958 |
  | cart15 | noncode | 2346 | 19 | 11615 | 20 | 0.9920 | 0.9915 | 0.9972 |

- **Verdict:** group prediction is **near-solved for every model** — group acc 0.989–0.995,
  per-group precision/recall ≥0.98, and 96–98% of class errors keep the true label inside the
  predicted group (val estimate, above the 92% error-analysis note in `src/data.py`). The whole
  ~24% class-error mass is intra-group disambiguation, dominated by the explore wing (§1: the
  synonymous file-ops). Implications: (a) a group-conditional / hierarchical head loses almost
  nothing at the group stage — the ceiling is intra-group; (b) any inference-time routing can
  trust the predicted group even at low MSP (complements §3, where MSP could not flag mislabels);
  (c) macro-F1 gains must come from separating explore-wing synonyms, not from group errors.
  4×4 group-confusion matrices per model: `group_confusion.py` output.

## Assets & reproduce

- Scripts (`experiments/coreset/`): `confident_learning.py` (C1, k-fold OOF, `--folds/--merge`
  for multi-GPU) · `score_dynamics.py` (C2/C4/C5/C6 from the dynamics npz) · `pvi.py` (C7, reuses
  C1's OOF) · `coreset_eval.py` (per-tier/per-class eval → `eval_results.md`) · `gate_analysis.py`
  · `group_confusion.py` (§5, CPU-only on cached logits) · `plot_drop_maps.py` · orchestrators
  `run_coreset*.py` (GPU-slot queue over the DAG).
- Plumbing (`src/finetune.py`): `--keep_indices <npy>` (absolute-index train filter) ·
  `--log_dynamics <npz>` (per-epoch train softmax callback).
- Caches: `analysis/cache/coreset_dyn_granite.npz` (dynamics E×N×C) · `coreset_oof_granite.npz`
  (4-fold OOF) · `coreset_gate_logits.npz` (gate-analysis val logits) · keep-sets in `keepsets/`
  (npy, gitignored — regenerate via the scorers). Logs: `sbatch/logs/coreset_*`, `e22_*_champion.log`.
