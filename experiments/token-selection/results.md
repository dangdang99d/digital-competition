# Branch: token-selection — upstream feature/token selection (E24)

Branch: `research/token-selection`. Scope: score **which parts of the input matter**, drop the
rest, retrain on the reduced input — feature selection as a **filter** (score once → retrain
once per kept-set), never a wrapper (no retrain-per-subset search). Decoupled from the model —
the deliberate contrast to embedded LTP (E18, deferred: in-model thresholds + a `1/T` gradient
wall). Methods: **A** token-level (attention/saliency top-k) · **B** field-level (occlusion) ·
D (controls) / C (rationale) gated on A/B showing slack.

**Baseline & metric:** deployment champion granite **E8a+LS richargs full_data = 0.7803** raw
uncal macro-F1 on the 3.5k held-out slice (seed 42) — also the frozen Phase-0 scorer and the
warm-start for recovery arms. Slice noise ≈ **±0.003** → |Δ|<0.003 = lossless. NO calibration.

**Phase-1 protocol:** every kept-set trains **two ways** under the LOCKED champion recipe (only
the input differs): **from-scratch** (HF base — the honest test) and **recovery** (warm-start
champion `checkpoint-8314`). Reduced inputs enter via `finetune.py --reduced_ids` (pre-tokenized
ids; split/recipe untouched). Faithfulness anchor: from-scratch on the FULL input must ≈ champion.

## Summary

| Method | Phase 0 (score) | Phase 1 (refit) | Result | Verdict |
|---|---|---|---|---|
| anchor | — | full input, from-scratch | **0.7790** ≈ champion 0.7803 | ✅ harness faithful |
| A1 attn-filter | champion attention received | k90 scratch + recover | **0.7712** / 0.7639 | ❌ −0.008 — 10% drop not free |
| A2 sal-filter | grad saliency `|∂L/∂emb·emb|` | k90 scratch | **0.7550** | ❌❌ −0.024 — worse than attn |
| B field-filter | leave-one-field-out occlusion (19 fields) | drop2/5/8 × both modes | drop5 scratch **0.7780**; drop2/8 0.7725 | 🟡 ~lossless, NO gain — efficiency only |
| A×B overlap | `a_drop_overlap.py` | — | attn∩sal ≈ chance (1.2× @k90) | no model-independent token set |
| D1 controls | random/stopword/truncate | — | not run | dropped — A/B null, nothing to control |
| C rationale | joint selector–predictor | — | not run | dropped — gated on slack; none found |

**VERDICT — selection is NULL for accuracy here; champion stays.** Our serialized inputs are
low-redundancy (as the E18 probe predicted): dropping even 10% of tokens costs real F1, and the
only safely-droppable structure is a handful of zero-signal meta subfields (shorter input =
efficiency, not accuracy). *Focus-on-the-Core*'s +5pt does not replicate. The k-sweep stop-rule
("descend only while recovery ≈ champion") fired at the first step — k80/70/60 never ran.

## A · Token selection (score → top-k keep → refit)

- **Change:** per example keep the top-k% highest-scoring content tokens (specials always kept,
  order preserved), k=90% first. Scores from the frozen champion (`a_score_tokens.py`, cached
  `artifacts/a_token_scores.npz`): **attn** = attention received (mean over layers/heads/non-pad
  queries) · **sal** = gradient saliency under true-label CE.
- **Result** (vs anchor 0.7790):

  | run | mode | val F1 | Δ anchor |
  |---|---|---|---|
  | attn k90 | scratch | 0.7712 | −0.008 |
  | attn k90 | recover | 0.7639 | (confounded — see findings) |
  | sal k90 | scratch | **0.7550** | **−0.024** |
- **Verdict:** ❌ token-drop hurts beyond noise at the *mildest* setting; saliency selection is
  markedly worse than attention. No free lunch, no denoise gain — line closed.

## B · Field selection (structured occlusion → drop-tail refit)

- **Phase 0** (training-free; frozen champion on the 3.5k slice; baseline replicated 0.7801;
  full ranking in `artifacts/b_field_importance.json`): importance ΔF1 when a field is occluded —
  **prompt +0.612 ≫ history_all +0.351 ≫ hist_depth≤4 +0.093 · meta_block +0.058 · action_bare
  +0.050 · action_detail +0.028 · meta.loc +0.026 · user_turns +0.025** … tail ≈ zero:
  meta.{open .010, ci .004, turn .004, dirty .002, budget .002, lang .001, tier .001, elapsed .001}.
  Prompt/history/action detail are load-bearing; most meta subfields are dead weight.
- **Phase 1** (nested tail drops; mean input 251→245/236/226 tok):

  | kept-set (dropped fields) | scratch | recover |
  |---|---|---|
  | drop2 (elapsed, tier) | 0.7725 | 0.7633 |
  | drop5 (+ lang, budget, dirty) | **0.7780** | 0.7623 |
  | drop8 (+ turn, ci, open) | 0.7725 | 0.7625 |
- **Verdict:** 🟡 dropping the ~zero-ΔF1 meta subfields is **lossless** (drop5 ≈ anchor; drop2/8
  −0.006, noise-dominated and non-monotonic) but yields **no accuracy gain** — structured
  denoising found no distractor fields to profit from. Usable as a free ~6–10% input-shortening.

## A×B · Scorer overlap — is there a transferable "unimportant token" set? (user ask 2026-07-09)

`a_drop_overlap.py` (CPU, from cached scores): per-sample Spearman(attn, sal) **0.35** median;
drop-set overlap **12.0% @k90 vs 9.7% chance (1.2×)**, 30.1% vs 19.8% @k80 (1.5×), 42.8% vs
29.7% @k70. attn drops by token *type* (serialization syntax — `␣'` 83% of its occurrences, `=`,
`_`, digits); sal drops by *context* (≤14% of any type, flat positional profile). The consensus ∩
(~1.2% of content tokens @k90) is almost pure serialization formatting — E21-consistent (format
isn't the edge). **No large model-independent unimportant-token set exists**; only the small
syntax core is a candidate ignore-set, and reusing it elsewhere needs an LB confirm against
train-overfit.

## Methodological findings (reusable)

1. **Recovery-degradation:** warm-starting the champion + 3 epochs @ lr 2e-5 on changed inputs
   lands ~0.763 *regardless of the input* — below every from-scratch twin. Recovery-vs-champion Δ
   is confounded; compare **from-scratch vs the from-scratch anchor**. (Recovery is NOT "strictly
   ≥ from-scratch" — contra the plan's assumption. Likely: 3 more epochs at full lr overshoots an
   already-converged model.)
2. **Faithfulness anchor works:** from-scratch full-input 0.7790 ≈ champion 0.7803 → reduced-input
   Δs are attributable to the input, not the harness.
3. **3.5k-slice noise ±0.003** — and it mis-ranks LB (E8 lesson): any borderline call needs a submit.

## Reproduce

- Scripts (`experiments/token-selection/`): `a_score_tokens.py` (Phase-0 A, ~10 min GPU) →
  `a_build_reduced.py --scorer attn|sal --keep 0.9…` (CPU) · `b_field_occlusion.py` (Phase-0 B,
  ~15 min GPU) → `b_build_reduced.py --drop <fields> --tag …` (CPU; transforms shared with
  Phase-0 via `b_fields.py`) · `a_drop_overlap.py` (CPU).
- Phase-1 = `python -m src.finetune` with the LOCKED constant block `--model ibm-granite/…-r2
  --serialize richargs --full_data --loss ls --label_smoothing 0.1 --epochs 3 --lr 2e-5
  --batch_size 4 --grad_accum 4 --max_len 512 --seed 42 --out_dir ./output/pat --results_name
  ft_results_e24.csv`, varying ONLY `--reduced_ids <npz>`, `--tag`, and `--init_from <champion>`
  (present = recovery, absent = scratch). ⚠️ `--reduced_ids` files must come from a FULL 70k
  Phase-0 run (a `--limit` smoke file indexes out of range).
- Artifacts (`artifacts/`, npz gitignored): `a_token_scores.npz` (~500 MB) · `a_reduced_*` /
  `b_reduced_*` kept-sets · `b_field_importance.json` (committed). Results CSV:
  `output/pat/ft_results_e24.csv`; logs `sbatch/logs/{a24,b24}_*.log`.

## References

- Input-length shortening via attention values — A1 recipe (arXiv 2303.07585)
- Focus on the Core: pruned token compression for document classification — B motivation (arXiv 2406.01283)
- Rationalizing Neural Predictions (arXiv 1606.04155) · HardKuma rationales (arXiv 1905.08160) — C (dropped)
- Learned Token Pruning (arXiv 2107.00910) — the embedded contrast (E18, deferred)
