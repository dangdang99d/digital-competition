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
| 0 | **screen**: harvest val logits of ~20 existing checkpoints on the shared 3.5k held-out → all-pairs + greedy (Caruana) uniform-softmax selection + disagreement/diversity analysis | ~20 fwd passes × 3.5k (≈1 h on 1 GPU; CPU for the math) | 🏃 running (GPUs 0–2; 13 pre-crash cached) |
| **1A** | **pair submission** (user-selected path 1): best granite pair → `build_submission.py` (shared keep-set prune + fp16 + shared remap/tokenizer + parity gate vs screen logits + zip) + `script_ensemble.py` (tokenize-once, length-sort, bs 256 fp16, sequential members, uniform prob mean) | CPU + 1 parity pass | 🔨 code ready |
| **1B** | **ensemble distillation** (user-selected path 2; = E10 revival): larger teacher (MAY include qwen3_ls — offline only) → `distill_teacher.py` (member train-logits → mean-softmax teacher npz) → from-scratch granite student via existing `finetune.py --distill_from/--distill_alpha/--distill_T` (CE+KD; LS not combinable) → normal single-model zip | ~1 fwd/member over 70k + 1.5 GPU-h/student | 🔨 code ready |
| 2 | submit best of 1A/1B → record LB + measured time (1A doubles as the fp16-pair timing datum) | 1–2 submission slots | 🔲 |
| opt | soup arm: 2–3 shared-init LR-diverse champions → uniform weight-merge (E12 revival, done right) | ~4.5 GPU-h | 🔲 propose separately |

Phase-0 screen exclusions (found in the partial pre-crash read): `e12_granite_ls_s43/s44`
(trained seed 43/44 → different full_data split → **leak** on our seed-42 slice: scored 0.7955 vs
own-val 0.7758) · `e4_ffn_r512` (factored FFN — plain `from_pretrained` loads garbage, 0.031).
Distillation expectations: students typically recover **60–90% of the ensemble−single gap** —
the teacher must clearly beat 0.7803 for 1B to be worth a slot.

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

## VRAM / batch-size sweep (measured 2026-07-10, RTX 3090; allocation bytes are GPU-independent → valid for T4)

Worst-case inputs (every row padded to 512). T4 allocator budget taken as **13.5 GiB**
(16 GB − CUDA context − margin). Current shipped zips run **bs=64** (granite fp32, qwen3 fp16).

**granite-311m (champion ckpt):** activations ≈ 7.6 MB/sample fp16 (ModernBERT local/global attention)

| dtype | weights | bs 64 | bs 128 | bs 256 | bs 384 | bs 512 |
|---|---|---|---|---|---|---|
| fp16 | 0.59 GiB | 1.08 | 1.57 | **2.54** | 3.51 | 4.48 — all fit T4 |
| fp32 | 1.20 GiB | 2.16 | 3.13 | 5.07 | 7.01 | 8.94 — all fit T4 |

**qwen3-0.6B (qwen3_ls ckpt):** activations ≈ 70 MB/sample fp16 (~9× granite — full S×S
attention, 28 layers, hidden 1024; no ModernBERT local-window discount)

| dtype | weights | bs 64 | bs 128 | bs 256 | bs 384 |
|---|---|---|---|---|---|
| fp16 | 1.11 GiB | 5.56 ✓ | **9.99 ✓** | 18.87 **OVER T4** | OOM on 24GB |
| fp32 | 2.23 GiB | 10.85 ✓ (tight) | 19.48 OVER | OOM | — |

**Verdict:** granite: memory nowhere near binding — **bs 256 fp16** shipped in
`script_ensemble.py` (512 provably safe; `ENS_BS` env override); the old bs-64 fp32 setup left
4× batch and 2× dtype on the table. qwen3: **bs 128 fp16 is the T4 ceiling** (9.99/13.5 GiB) —
only a 2× batch bump over its shipped bs 64, and a 0.6B full-attention model is likely
compute-bound on a T4 anyway → the 9:18 wall probably moves little; granite+qwen3 pairs stay
dead unless a re-timed submission proves otherwise. qwen3-depth14 (~half the layers ≈ ~35
MB/sample) should take bs 256 — relevant only if the depth-14 pair idea ever earns a slot.
Speed impact is a T4-throughput question only a real submission measures. Logs:
`sbatch/logs/e26_bs_sweep{,_qwen}.log`.

## Combiner taxonomy — ALL cheap ones evaluated (user 2026-07-10; `combiners.py`, CPU over cached slice logits)

| combiner | fitted params | note |
|---|---|---|
| **soft-uniform** (mean of member softmax) | 0 | the DEFAULT (ships unless beaten honestly); prob-space neutralizes LS-vs-CE logit temperature |
| logit-mean | 0 | scale-sensitive: implicitly overweights sharp (CE) members |
| geo-mean (product of experts) | 0 | punishes any member's low prob — stricter consensus |
| hard-vote (majority) | 0 | discards confidence; ties broken by soft-uniform |
| rank-mean | 0 | scale-free; coarse at C=14 |
| weighted prob mean ⚠️ | M | fit on slice = calibration-adjacent + E8 mis-rank risk → reported as 2-fold honest CV **and** full-slice upper bound; shipping it needs explicit user sign-off |
| stacking (LR on member probs) ⚠️ | M·C·C | max capacity, same objections; 2-fold honest CV only |
| weight-space soup / SWA | 0 | NOT logit-level — needs shared-init retrains (E12); optional arm |
| distillation (Path 1B) | — | ensemble → single model; the constraint-dissolving variant |

## Results

*(pending Phase 0)*

## Reproduce

- `experiments/ensemble/screen_ensemble.py` — Phase-0 harvester + greedy selection (logits cached
  `analysis/cache/e26_screen_logits.npz`, idempotent; `--limit` smoke).
- Phase-1 packaging: reuse the granite zip build (`submission-build-recipe` memory,
  `submit_0707_granite_ls_fp16.zip` as the fp16 reference) + `src/prune_vocab.py` for the
  granite-richargs prune; parity gate vs each member's own val logits before zipping.
