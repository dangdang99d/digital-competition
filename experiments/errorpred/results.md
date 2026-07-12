# E27 · Error prediction — can we tell WHICH rows the model will get wrong, and recover them?

Objective (user 2026-07-09): a **label-free, test-applicable** per-row estimate of "this
prediction is likely wrong", plus structure that hints at the true label on flagged rows.
Anchors: E20 (~0.85 MSP gate ceiling, model-independent) · E22-§5 (group acc ~0.99) ·
E6 (blind top-2 fallback net-negative). All raw fp32 logits, NO calibration.

Analysis model = **granite-LS e9** (standard split, val out-of-sample; the deployed
e8a_ls champion is full_data-trained → its val logits are in-sample and unusable).
Logits cached: `analysis/cache/clpvi_tiers_granite_ls.npz`.

## Phase 1 · MSP works as the error estimate; the truth hides in the top-4

`decile_confusion.py` — rank val rows by MSP (ascending deciles), then per decile:
confusion matrices for the counterfactual "pick rank-k" (k=1..4) + where the true label
sits in the model's ranking. AUROC(MSP→own error) = **0.843**; error rate runs
63.6% (bottom decile) → 1.4–1.8% (top deciles) vs 24.2% base.

| decile (MSP asc) | MSP range | acc@1 | P(true=r2\|wrong) | P(true=r3\|wrong) | top2 | top3 | top4 | P(true∈r1 grp) | …\|wrong |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.18–0.42 | 0.364 | 0.508 | 0.280 | 0.687 | 0.865 | 0.986 | 0.978 | 0.965 |
| 2 | 0.42–0.51 | 0.429 | 0.576 | 0.248 | 0.758 | 0.899 | 0.996 | 0.992 | 0.986 |
| 3 | 0.51–0.64 | 0.507 | 0.628 | 0.222 | 0.816 | 0.926 | 0.997 | 0.991 | 0.983 |
| 4 | 0.64–0.76 | 0.724 | 0.627 | 0.244 | 0.897 | 0.964 | 0.994 | 0.994 | 0.977 |
| 5 | 0.76–0.82 | 0.836 | 0.533 | 0.328 | 0.924 | 0.977 | 0.996 | 0.996 | 0.978 |
| 6 | 0.82–0.86 | 0.837 | 0.570 | 0.281 | 0.930 | 0.976 | 0.996 | 0.997 | 0.982 |
| 7 | 0.86–0.90 | 0.928 | 0.604 | 0.257 | 0.971 | 0.990 | 0.998 | 0.997 | 0.960 |
| 8 | 0.90–0.91 | 0.985 | 0.571 | 0.333 | 0.994 | 0.999 | 0.999 | 0.999 | 0.952 |
| 9 | 0.91–0.91 | 0.986 | 0.650 | 0.200 | 0.995 | 0.998 | 0.999 | 1.000 | 1.000 |
| 10 | 0.91–0.98 | 0.982 | 0.440 | 0.440 | 0.990 | 0.998 | 0.999 | 1.000 | 1.000 |

Rank-2 group membership (does the runner-up stay in rank-1's action group?):

| decile | P(r2 in r1 grp) | P(true=r2 \| wrong & r2 in grp) | P(true=r2 \| wrong & r2 out grp) |
|---|---|---|---|
| 1 | 0.977 | 0.514 | 0.273 |
| 2–6 | 0.98–0.99 | 0.54–0.64 | ~0 (tiny n) |
| 7 | 0.804 | 0.621 | 0.333 |
| 8–10 | 0.35–0.41 | 0.50–0.81 | ~0 |

**Findings**

1. **The true label lives in the model's top-4 ≥98.6% of the time in EVERY decile** — even
   at MSP 0.18–0.42 where top-1 is wrong 64% of the time. Rank-5+ is ~1% everywhere.
   Combined with group containment ≥0.95 among wrong rows: an error is a *permutation
   within the predicted group*, essentially never an escape from it (confirms §5 at
   decile granularity; visible as block-diagonal structure in the rank-2 heatmaps for
   deciles 1–6).
2. **Blind rank-swapping is dead, now quantified** (E6's failure explained): even in the
   worst decile, keeping rank-1 scores 36.4% vs swapping-to-rank-2 32.3%. Rank-1 beats
   rank-2 conditional on ANY confidence level → no permutation of the model's own ranking
   helps. An exploit must inject an **independent signal** to rerank the top-2/3 candidates
   on low-MSP rows (second model, TTA vote, group-specialist, priors).
3. In confident deciles (8–10) the runner-up leaves the group (r2 in-group only 35–41%)
   but it's irrelevant — acc@1 is 98%+ there.
4. Low-decile confusion is exactly the §1 explore wing (read/grep/lsdir/glob block) plus
   bash↔tests↔lint and ask↔plan↔web.

5. **Per-true-class rank profiles** ([by_class fig](figures/true_rank_profile_by_class.png),
   `class_rank_profile.py`): the explore wing splits asymmetrically — when the model is
   unconfident, true **read/grep** sit at rank-2 (pairwise swaps), but true **glob/lsdir**
   get BURIED at rank-3/4 behind both read and grep (glob deciles 1–3: rank-1 only 10–28%,
   truth mostly rank 3–4). The intra-group ordering tracks class frequency (read 1851 /
   grep 1982 ≫ glob 1057 / lsdir 866 on val) — a prior-dominance bias, so any intra-group
   rescue must push *against* the majority sibling, not just pick rank-2. Class placement
   also differs wildly: edit/write/resp live almost entirely in deciles 7–10 (easy),
   read/grep/lsdir/glob dominate deciles 1–3. ⚠️ true-class grouping is DIAGNOSTIC ONLY —
   not observable at inference.
6. **Inference-accessible view** ([by_predclass fig](figures/true_rank_profile_by_predclass.png),
   `pred_class_rank_profile.py`): same profile grouped by (PREDICTED class × decile) — both
   observable at test time. Readable there: `lsdir` predictions are NEVER confident (all 956
   rows in deciles 1–4; cell acc 42–50%); predicted grep/read/glob in deciles 1–3 carry
   30–35% truth-at-rank-2 mass; predicted plan/ask low-deciles similar. These cells are the
   map of WHERE an external rescue signal should be spent — but rank_policy (1b) shows the
   model's own ranking can't cash them in.

Figures: [true_rank_profile](figures/true_rank_profile.png) ·
[by_class](figures/true_rank_profile_by_class.png) ·
[by_predclass 20-bin](figures/true_rank_profile_by_predclass_b20.png) ·
[rank1](figures/rank1_confusion_by_decile.png) · [rank2](figures/rank2_confusion_by_decile.png) ·
[rank3](figures/rank3_confusion_by_decile.png) · [rank4](figures/rank4_confusion_by_decile.png)

## Phase 1b · Rank-selection policy per (decile × predicted class) — NULL, confirms E6

`rank_policy.py` — for each cell (MSP decile × rank-1 predicted class; observables only),
pick the rank k∈{1..4} with the highest empirical P(true = rank-k), n≥20 guard; evaluate
in-sample AND 2-fold cross-fitted (fit cells on one val half, apply to the other).

- **In-sample:** only 6/140 cells prefer rank-2 (grep d1–d2, plan d1–d2, bash d2, ask d2;
  Δ +0.010–0.082 per cell) → policy acc 0.7600 vs base 0.7579 (+0.0021), mF1 +0.0039.
- **Cross-fitted (honest):** acc **−0.0020**, macro-F1 **−0.0013** — the winning cells don't
  replicate across halves; folds pick different cells (4 vs 5). The in-sample gain is noise
  mining. Even the largest cell (grep, decile 1, n=511) is rank1 0.323 vs rank2 0.348 — a
  2.5pp edge too thin to survive.
- **Verdict:** there is NO (decile × predicted-class) cell where selecting a lower rank
  reliably beats the model's own argmax — third independent confirmation (E6 fallback,
  E27-1 blind swap, now conditioned policy). Rank permutation from the model's own outputs
  is CLOSED; recovery requires an external signal (second model / TTA / priors).

## Phase 1c · (rank1, rank2, decile) pair statistics — first real FLIP cells

`pair_rank_stats.py` — condition on the full observable triple (rank-1 class, rank-2 class,
MSP decile): P(true=r1) vs P(true=r2) per cell. Figures:
[pair_flip_heatmap](figures/pair_flip_heatmap.png) ·
[pair_decile_profiles](figures/pair_decile_profiles.png).

- Aggregated over deciles 1–5 the heatmap is all-red (rank-1 safe in every pair, n≥30) —
  but **decile-sliced cells DO flip**, and they're coherent, not scattered:

  | rank1 | rank2 | decile | n | P(true=r1) | P(true=r2) |
  |---|---|---|---|---|---|
  | read | lsdir | 1 | 85 | 0.271 | **0.447** |
  | grep | lsdir | 1 | 84 | 0.250 | **0.464** |
  | bash | tests | 2 | 22 | 0.364 | **0.636** |
  | plan | ask | 2 | 73 | 0.425 | **0.493** |
  | grep | read | 1–2 | 389/356 | 0.33/0.38 | ≈tie 0.34/0.39 |

- **The flips are exactly the buried-minority pattern from the by-class profiles:** when an
  unconfident majority-class prediction (read/grep) has minority sibling **lsdir** as
  runner-up, the runner-up is nearly 2× more likely correct. The model's prior-dominance
  bias is *visible from observables* via the rank-2 identity — this is what the 1b policy
  (conditioned on r1 only) diluted away.
- ⚠️ In-sample cells, borderline per-cell significance (Δ≈0.2 at n≈85 ≈ 2.4σ) with heavy
  multiple-comparisons exposure (14×14×10 cells) — same trap class as 1b → verified below.

**1c-verify (`pair_policy_check.py`, user GO 2026-07-10)** — two checks, 7 models
(granite_ls e9 · 5 coreset models · qwen3; deciles = each model's own MSP quantiles):

- **Replication:** the lsdir-runner-up flips are REAL — `read→lsdir d1` replicates in 5/7
  models, `grep→lsdir d1` in 5/6 (only `cl`, the overconfidence-broken model, misses both);
  `grep→read d1` is a genuine tie in 6/7. But `plan→ask d2` replicates in 1/7 (noise —
  killed) and `bash→tests d2` is mixed (3/6, thin n).
- **Cross-fitted swap policy** (2-fold × 5 seeds, swap cells n≥20 where r2 wins in the fit
  half): unlike 1b, this survives honest evaluation on most models —
  granite_ls **ΔmF1 +0.0021** (positive on all 5 seeds; Δacc +0.0005) · base +0.0021/+0.0020 ·
  qwen3 +0.0020/+0.0021 · cart15 +0.0053 · aum06 ≈0 · pvi06 −0.0012 · cl ≈0.
  Conditioning on the rank-2 identity is what 1b was missing — it isolates the
  prior-dominance cells instead of averaging them into (decile × r1) cells.
- **Verdict:** first observable-only rule with a replicated positive effect — but small
  (≈+0.002 macro-F1 on val) and val-only.

## Phase 1d · Final flip rule + deployment transfer + variants (user GO 2026-07-10)

- **Full-fit rule** (`flip_rule_final.py`, all 14k, n≥20): **8 cells** → artifact
  `flip_cells.json` (read→lsdir d1 · grep→{read d1,d2 · lsdir d1 · glob d2} · bash→tests d2 ·
  ask→plan d2 · plan→ask d2). e9 in-sample mF1 0.7557→0.7611 (upper bound, ignore).
- **Honest transfer, 3.5k full_data held-out** (rule fixed, self-quantile deciles,
  e26-cached logits): **e8a_ls (deployment champion) mF1 0.7801→0.7813 (+0.0012)**, 129 rows
  swapped, acc unchanged · e8b_ls qwen3 +0.0007. Positive, small.
- **Group-restricted r2 variant** (`variant_group_r2.py`, user ask — mask non-group classes
  before the runner-up): cells ≈ identical (low-decile raw r2 is already 98–99% in-group),
  cross-fit numbers match raw, e8a transfer +0.0001 — **no advantage; raw-r2 rule stands**.
  Out-of-group r2s cost coverage in principle, but the affected rows are too few to matter.
- **Margin threshold (user Q, 2026-07-10):** requiring P(true=r2)−P(true=r1) > margin before
  a cell qualifies — margin 0/0.03/0.05/0.10 → e9 crossfit ΔmF1 +0.0021/+0.0019/+0.0020/+0.0025
  (flat), e8a 3.5k transfer +0.0012/+0.0004/+0.0004/−0.0005 (129/19/19/15 swaps). No margin
  helps; the big-n tie cells (grep→read) carried most of the e8a transfer despite tiny Δ
  (macro-F1 reallocation), and all deltas are within slice noise → **keep margin 0**, LB decides.
- **Group conditioning of r2 (user Q):** within a decile, r2-out-of-group rows behave
  differently on BOTH ends — low deciles: out-group r2 is rare (≤2%) and its truth is rarely
  at r2 (0.27 vs 0.51 in-group) or r3 → never swap those; high deciles (7–10): out-group r2
  becomes the majority (59–65%) and **acc@1 is higher when r2 is out-of-group (0.995–0.996 vs
  0.961–0.971 in-group)** — no in-group competitor = clean decision. "r2 in group(r1)" is
  itself a risk signal; candidate assessor feature.
- **Submission for LB validation — ✅ BUILT `submit_0710_flip.zip`** (20-char name per the new
  32-char filename limit): clone of `submit_0707_granite_ls.zip` (plain champion, LB 0.77738 —
  clean attribution) + the 8-cell rule as a discrete post-decision in `script.py` (raw logits
  untouched; self-quantile deciles; rule skipped if <500 rows). Validated WITHOUT GPU (user
  constraint): flip block exec'd verbatim on cached e8a logits == numpy reference (129/129
  swaps, predictions identical) · 5-row CPU end-to-end pass (model load, serialization,
  skip-path, CSV) · zip diff vs original: all 7 files byte-identical except script.py,
  integrity OK, 846M. Ready to upload; read LB vs 0.77738.
- **LB RESULT (submitted 2026-07-10 19:05): 0.77727 = −0.00011 vs champion 0.77738 —
  NEUTRAL; the +0.0012 slice gain did NOT transfer** (5:06, rule costs zero time).
  Consistent with the information bound: the fixable residue in the model's own outputs
  is ≈0 on the real test distribution.

## VERDICT — E27 CLOSED (user decision + LB read, 2026-07-10)

Detect-and-fix on top of a trained classifier is information-bounded: every adjustment
reading only the model's own logits failed honest evaluation (E6 fallback · blind swap ·
1b policy · flip rule LB-neutral) — model+rule is just another classifier that training
could have learned. Value enters only with OUTSIDE information, and each such source is
better spent at full coverage than through a gate: cross-model signal → E26 ensembling
(LB +0.0079) dominates selective consultation; test-distribution signal → TTA (+0.0019).
Durable outputs: top-4/group containment map · prior-dominance (buried-lsdir) bias ·
assessor proof that cross-model features break the 0.85 ceiling (0.8655 vs 0.8425) —
carried into E26 as guidance, not as a routing layer.

## Phase 2-B · Assessor v2 — CROSS-MODEL FEATURES BREAK THE 0.85 CEILING ✅

`assessor_b.py` (user GO 2026-07-10) — HistGradientBoosting, 5-fold CV, target =
e9 granite-LS error, all features label-free/test-computable. No embeddings yet
(kNN-difficulty block deferred, needs GPU pass).

| feature set | dims | AUROC |
|---|---|---|
| MSP alone | 1 | 0.8425 ± 0.0072 |
| OWN logit shape (top-4 probs, entropy, margins, group-margin, r1/r2 one-hots) | 36 | 0.8422 ± 0.0069 |
| **OWN + CROSS** (6 helpers' MSP · agreement · pairwise KL · agree-count) | 55 | **0.8655 ± 0.0044** |

- **First method on this project to beat the E20 MSP ceiling** (+0.023 AUROC), and via
  exactly the literature-predicted mechanism: cross-model disagreement is information the
  model's own logits provably lack (OWN ties MSP at 0.842 — E20 re-confirmed a 4th time).
- Error capture at fixed flag budgets (vs MSP): 10% → 28.8% vs 26.3% · 20% → 53.1% vs
  49.8% · 25% → 64.1% vs 61.1%.
- Deployment cost caveat: CROSS features need ≥2 models at inference (granite pair fits
  the 10-min budget per E26: 4:30). Next options: add kNN-difficulty features (GPU
  embedding pass) · target the deployment model on the 3.5k · wire assessor→action
  (selective TTA / selective ensemble / flip-cells on flagged rows).

## Phase 2 · Improving the estimator — literature survey (3 agents, 2026-07-09)

Post-hoc scores vs MSP: **globally replicated dead end** (FD-Shifts ICLR'23; Zhu TPAMI'24;
Galil ICLR'23 — nothing post-hoc beats MSP on well-trained in-distribution classifiers;
GradNorm/energy/ODIN/conformal-set-size/TrustScore/MC-dropout all confirmed ≤ MSP).
Live candidates:

- **A · LS-damage audit + p-norm-logit rescue (zero GPU).** LS training documented to cost
  3–9 AUROC pts of error-ranking (Zhu ECCV'22; Xia & Bouganis ICLR'25 — LS suppresses the
  max logit MORE on correct rows); our champion & granite-LS are LS ε=0.1. Post-hoc fix =
  score by max of p-norm-normalized logits (keeps the model + accuracy). Test on cached
  LS-vs-CE twin logits (e9_granite_ls vs coreset CE base), + Rel-U (ICLR'24) same pass.
- **B · Assessor v2 (CPU/minor GPU).** XGBoost on features MSP lacks: 6 cached models'
  MSPs/agreement/pairwise-KL (cross-model), group-restricted margin (§5), kNN-propagated
  CL/PVI difficulty from labeled val. Trained calibrators beat MSP only with such extra
  features or under shift (Kamath ACL'20; Varshney'22) — and E23 proved our test HAS
  exploitable shift.
- **C · CRL retrain** (ICML'20; text-validated ACL'21): rank confidence by training-history
  correctness; +3.5–6.7 AUROC reported; replaces LS.
- **D · SWA/SAM flat-minima retrain** (FMFP ECCV'22): largest reported gains (+4–7 AUROC),
  MSP unchanged at inference.

- **Status:** 🔵 ACTIVE — phase 1 done; phase 2 arm awaiting user pick.
