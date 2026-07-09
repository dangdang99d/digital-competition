# Branch: ensemble (E26) — combine existing checkpoints for the highest submittable score

Logical group `research/ensemble` (committed on `research/token-selection`). Objective: the
**highest-performing SUBMITTABLE model** by ensembling checkpoints we already trained — select
members from previous experiments, combine, package under the submission constraints. All scores
**raw uncalibrated macro-F1**; combination = **plain uniform mean of softmax** across members —
NO per-class weights, NO val-tuned combination weights (calibration-adjacent + 3.5k-overfit;
project invariant).

Baselines to beat: **LB 0.77931** (granite+TTA, current SOTA) · 0.77921 (qwen3_ls, best non-TTA)
· held-out screen reference **0.7803** (granite champion on the shared 3.5k slice).

## Submission constraints (the binding design input — measured anchors ONLY, never local projection)

| constraint | value | measured anchors (LB server, T4 16GB · 3 vCPU · 12GB RAM · offline) |
|---|---|---|
| zip size | **≤ 1 GB** | granite fp32 846M · granite fp16 539M · qwen3_ls (fp16+vocab-prune) 830M · qwen3 depth14 443M |
| inference | **≤ 10 min** / 30k rows | granite fp32 **5:06** · qwen3_ls **9:18** · TTA adds **+2:40** · granite fp16 / qwen3-depth14: **unmeasured** |

Feasibility math (why most ensembles are dead on arrival) — user-confirmed 2026-07-09:
- **anything containing full qwen3 is DEAD** — 9:18 alone; no second forward fits (user agreed).
- **2× granite fp32 ≈ 10:12 → over.** The viable shape is **2× granite fp16 + vocab-prune**.
  ⚠️ fp16, NOT bf16: the eval T4 is Turing — no bf16 support (bf16 is our 3090 *training*
  format only). fp16 granite is parity-validated (`submit_0707_granite_ls_fp16.zip`, 539M).
  Size: 2× 539M zips ≈ 1078M > 1GB → **granite vocab-prune required** (embeddings = ~62% of
  granite-311m; prune → est. ~250–350M/model → pair ≈ 600–700M ✓; infra `src/prune_vocab.py`,
  tokenizer+remap reusable per backbone+variant per memory).
- **Batch-size lever (user note):** submission inference under-utilizes the T4 — raise the eval
  batch size (fp16 @512 on 16GB has ample headroom) + keep length-sorted batching. Free speed
  for the second forward pass; validate locally for OOM only (time is server-measured).
- **weight-space soup / SWA is constraint-FREE** (one merged model = champion cost: 5:06/846M) —
  but E12 showed soup members MUST share init (different-init soup cratered −0.056); a proper
  soup arm needs 2–3 new shared-init LR-diverse trainings (~1.5 GPU-h each). Optional arm.
- **TTA: excluded from E26** (user 2026-07-09) — it adapts each member separately (+2:40 *per
  model*), and a two-forward budget has no slack; granite+TTA (0.77931) stays a separate
  single-model submission line, and E26 must beat it to matter.

## Plan

| Phase | What | Cost | Status |
|---|---|---|:--:|
| 0 | **screen**: harvest val logits of ~20 existing checkpoints on the shared 3.5k held-out → all-pairs + greedy (Caruana) uniform-softmax selection + disagreement/diversity analysis | ~20 fwd passes × 3.5k (≈1 h on 1 GPU; CPU for the math) | 🔲 built (`screen_ensemble.py`), awaiting go |
| 1 | package the best constraint-feasible combo: fp16 + granite vocab-prune + parity gate + zip | CPU + 1 parity pass | 🔲 gated on Phase 0 ≥ champion +0.003 |
| 2 | submit → record LB score + measured time (the fp16-pair timing datum) | 1 submission slot | 🔲 |
| opt | soup arm: 2–3 shared-init LR-diverse champions → uniform weight-merge (E12 revival, done right) | ~4.5 GPU-h | 🔲 propose separately |

**Decision gates:** package only if the screen beats the champion **0.7803 by > +0.003** (slice
noise); prefer *diverse* members at equal F1 (different recipe/serialization/data — disagreement
is where ensemble gains live). E8 lesson stands: the 3.5k slice mis-ranks LB — the screen picks
the candidate, **LB is the judge**; keep the first submission conservative (2 members).

## Phase 0 candidate pool (existing checkpoints, `output/pat/`; screen re-measures all uniformly)

Eval slice: the shared 3.5k held-out (full_data seed-42 `va_eval`) — every full_data model never
saw it; standard-split models never saw any of the 14k val ⊃ it. Serialization per model comes
from its `ft_results*.csv` row (v1 / richargs / richmeta / names / miseo).

| family | members (known 3.5k-slice or own-val F1) |
|---|---|
| granite richargs full_data | **e8a_ls 0.7803 (champion)** · a24_anchor_full_scratch 0.7790 (same-recipe reroll) · e22_aum06 0.7766 · e22_pvi06 0.7700 · estack_v2 (TAPT+LS) 0.7769 · estack v1 0.772 · e8a (CE) · e12_ls_s43/s44 · e12soup_is2/is3 · e16b_depth11/14 (pruned) |
| granite, other serializations | e25c_richmeta_ls 0.7801 · granite_names_ls 0.7731 · e25a_miseo 0.7692 · e25b_miseo_ls |
| granite v1 standard-split | e9_ls 0.7565 · e11_tapt 0.7592 · e9 {ce, wce, la, focal} · coreset_rt_pvi06/aum06 · coreset_base |
| qwen3 (screen for info; full qwen3 unpackageable) | e8b_ls 0.7659-slice/**LB 0.77921** · e8b (CE) · e4_ffn_r512 0.7688 · e16_depth14 0.7638 (443M, ~½ time — the only *possibly* packageable qwen3) |
| excluded | a24/b24 reduced-input models (custom input pipeline; add later via `--reduced_ids` val rebuild if wanted) · bge-m3 specialists (`ceil_*`) · smoke runs · bge-m3 richargs (weights only inside old zips) |

**Why the pool is promising despite same-backbone dominance:** we hold genuinely diverse axes —
recipe (CE/LS/TAPT/miseo), serialization (v1/richargs/richmeta/names), data (full vs denoised vs
field-dropped), depth (pruned), and backbone (qwen3-depth14). The screen's disagreement matrix
tells us which axes actually decorrelate errors; §5 of coreset/results.md (errors ~all
intra-group) predicts serialization/recipe diversity matters more than backbone.

## Results

*(pending Phase 0)*

## Reproduce

- `experiments/ensemble/screen_ensemble.py` — Phase-0 harvester + greedy selection (logits cached
  `analysis/cache/e26_screen_logits.npz`, idempotent; `--limit` smoke).
- Phase-1 packaging: reuse the granite zip build (`submission-build-recipe` memory,
  `submit_0707_granite_ls_fp16.zip` as the fp16 reference) + `src/prune_vocab.py` for the
  granite-richargs prune; parity gate vs each member's own val logits before zipping.
