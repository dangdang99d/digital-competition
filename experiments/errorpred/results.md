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

Figures: [true_rank_profile](figures/true_rank_profile.png) ·
[rank1](figures/rank1_confusion_by_decile.png) · [rank2](figures/rank2_confusion_by_decile.png) ·
[rank3](figures/rank3_confusion_by_decile.png) · [rank4](figures/rank4_confusion_by_decile.png)

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
