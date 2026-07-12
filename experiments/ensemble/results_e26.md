# Branch: ensemble (E26) — combine existing checkpoints for the highest submittable score

**STATUS: ✅ DONE (2026-07-12).** Outcome: **trio ensemble = LB 0.78719, current SOTA**
(is3 + aum06 + e25c, uniform prob mean, fp16 bs256, 6:52/836M — the direct-package ceiling;
a 4th member breaks both caps). Path-1A pairs ✅ (0.78548/0.78498) · path-1B distillation
❌ CLOSED (in-sample teachers; all students below champion) · combiner shoot-out: uniform
mean unbeaten (every fitted combiner lost honest CV). Continuation = **E30** (session-grouped
OOF campaign: honest teacher retry + E29 learned-head data) — running 2026-07-12.

Logical group `research/ensemble` (committed on `research/token-selection`). Objective: the
**highest-performing SUBMITTABLE model** by ensembling checkpoints we already trained — select
members from previous experiments, combine, package under the submission constraints. All scores
**raw uncalibrated macro-F1**; combination = **plain uniform mean of softmax** across members —
NO per-class weights, NO val-tuned combination weights (calibration-adjacent + 3.5k-overfit;
project invariant).

Baselines to beat: **LB 0.77931** (granite+TTA, current SOTA) · 0.77921 (qwen3_ls, best non-TTA)
· held-out screen reference **0.7803** (granite champion on the shared 3.5k slice).

## Submission constraints (the binding design input — measured anchors ONLY, never local projection)

| constraint | value                   | measured anchors (LB server, T4 16GB · 3 vCPU · 12GB RAM · offline)                                           |
| ---------- | ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| zip size   | **≤ 1 GB**              | granite fp32 846M · granite fp16 539M · qwen3_ls (fp16+vocab-prune) 830M · qwen3 depth14 443M                 |
| inference  | **≤ 10 min** / 30k rows | granite fp32 **5:06** · qwen3_ls **9:18** · TTA adds **+2:40** · granite fp16 / qwen3-depth14: **unmeasured** |

Feasibility math (why most ensembles are dead on arrival) — user-confirmed 2026-07-09:

- **anything containing full qwen3 is DEAD** — 9:18 alone; no second forward fits (user agreed).
- **2× granite fp32 ≈ 10:12 → over.** The viable shape is **2× granite fp16 + vocab-prune**.
  ⚠️ fp16, NOT bf16: the eval T4 is Turing — no bf16 support (bf16 is our 3090 _training_
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
- **TTA: excluded from E26** (user 2026-07-09) — it adapts each member separately (+2:40 _per
  model_), and a two-forward budget has no slack; granite+TTA (0.77931) stays a separate
  single-model submission line, and E26 must beat it to matter.

## Plan

| Phase  | What                                                                                                                                                                                                                                                                                                                                                    | Cost                                                    |                                                       Status                                                        |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | :-----------------------------------------------------------------------------------------------------------------: |
| 0      | **screen**: harvest val logits of ~20 existing checkpoints on the shared 3.5k held-out → all-pairs + greedy (Caruana) uniform-softmax selection + disagreement/diversity analysis                                                                                                                                                                       | ~20 fwd passes × 3.5k (≈1 h on 1 GPU; CPU for the math) |                           ✅ 26/34 cached (all heavy hitters; 8 stragglers never needed)                            |
| **1A** | **pair submission** (user-selected path 1): best granite pair → `build_submission.py` (shared keep-set prune + fp16 + shared remap/tokenizer + parity gate vs screen logits + zip) + `script_ensemble.py` (tokenize-once, length-sort, bs 256 fp16, sequential members, uniform prob mean)                                                              | CPU + 1 parity pass                                     |                                      ✅ pairs SUBMITTED → LB 0.78548 / 0.78498                                      |
| **1B** | **ensemble distillation** (user-selected path 2; = E10 revival): larger teacher (MAY include qwen3_ls — offline only) → `distill_teacher.py` (member train-logits → mean-softmax teacher npz) → from-scratch granite student via existing `finetune.py --distill_from/--distill_alpha/--distill_T` (CE+KD; LS not combinable) → normal single-model zip | ~1 fwd/member over 70k + 1.5 GPU-h/student              | ❌ CLOSED — all 4 students 0.7635–0.7728 < champion (in-sample teachers; see verdict below). Honest retry = **E30** |
| 2      | submit best of 1A/1B → record LB + measured time (1A doubles as the fp16-pair timing datum)                                                                                                                                                                                                                                                             | 1–2 submission slots                                    |                              ✅ pairs + trio submitted → **trio 0.78719 SOTA** (6:52)                               |
| opt    | soup arm: 2–3 shared-init LR-diverse champions → uniform weight-merge (E12 revival, done right)                                                                                                                                                                                                                                                         | ~4.5 GPU-h                                              |                                  ⛔ never run — superseded by the trio + E30 line                                   |

Phase-0 screen exclusions (found in the partial pre-crash read): `e12_granite_ls_s43/s44`
(trained seed 43/44 → different full_data split → **leak** on our seed-42 slice: scored 0.7955 vs
own-val 0.7758) · `e4_ffn_r512` (factored FFN — plain `from_pretrained` loads garbage, 0.031).
Distillation expectations: students typically recover **60–90% of the ensemble−single gap** —
the teacher must clearly beat 0.7803 for 1B to be worth a slot.

**Decision gates:** package only if the screen beats the champion **0.7803 by > +0.003** (slice
noise); prefer _diverse_ members at equal F1 (different recipe/serialization/data — disagreement
is where ensemble gains live). E8 lesson stands: the 3.5k slice mis-ranks LB — the screen picks
the candidate, **LB is the judge**; keep the first submission conservative (2 members).

## Phase 0 candidate pool (existing checkpoints, `output/pat/`; screen re-measures all uniformly)

Eval slice: the shared 3.5k held-out (full_data seed-42 `va_eval`) — every full_data model never
saw it; standard-split models never saw any of the 14k val ⊃ it. Serialization per model comes
from its `ft_results*.csv` row (v1 / richargs / richmeta / names / miseo).

| family                                            | members (known 3.5k-slice or own-val F1)                                                                                                                                                                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| granite richargs full_data                        | **e8a_ls 0.7803 (champion)** · a24_anchor_full_scratch 0.7790 (same-recipe reroll) · e22_aum06 0.7766 · e22_pvi06 0.7700 · estack_v2 (TAPT+LS) 0.7769 · estack v1 0.772 · e8a (CE) · e12_ls_s43/s44 · e12soup_is2/is3 · e16b_depth11/14 (pruned) |
| granite, other serializations                     | e25c_richmeta_ls 0.7801 · granite_names_ls 0.7731 · e25a_miseo 0.7692 · e25b_miseo_ls                                                                                                                                                            |
| granite v1 standard-split                         | e9_ls 0.7565 · e11_tapt 0.7592 · e9 {ce, wce, la, focal} · coreset_rt_pvi06/aum06 · coreset_base                                                                                                                                                 |
| qwen3 (screen for info; full qwen3 unpackageable) | e8b_ls 0.7659-slice/**LB 0.77921** · e8b (CE) · e4_ffn_r512 0.7688 · e16_depth14 0.7638 (443M, ~½ time — the only _possibly_ packageable qwen3)                                                                                                  |
| excluded                                          | a24/b24 reduced-input models (custom input pipeline; add later via `--reduced_ids` val rebuild if wanted) · bge-m3 specialists (`ceil_*`) · smoke runs · bge-m3 richargs (weights only inside old zips)                                          |

**Why the pool is promising despite same-backbone dominance:** we hold genuinely diverse axes —
recipe (CE/LS/TAPT/miseo), serialization (v1/richargs/richmeta/names), data (full vs denoised vs
field-dropped), depth (pruned), and backbone (qwen3-depth14). The screen's disagreement matrix
tells us which axes actually decorrelate errors; §5 of coreset/results.md (errors ~all
intra-group) predicts serialization/recipe diversity matters more than backbone.

## VRAM / batch-size sweep (measured 2026-07-10, RTX 3090; allocation bytes are GPU-independent → valid for T4)

Worst-case inputs (every row padded to 512). T4 allocator budget taken as **13.5 GiB**
(16 GB − CUDA context − margin). Current shipped zips run **bs=64** (granite fp32, qwen3 fp16).

**granite-311m (champion ckpt):** activations ≈ 7.6 MB/sample fp16 (ModernBERT local/global attention)

| dtype | weights  | bs 64 | bs 128 | bs 256   | bs 384 | bs 512            |
| ----- | -------- | ----- | ------ | -------- | ------ | ----------------- |
| fp16  | 0.59 GiB | 1.08  | 1.57   | **2.54** | 3.51   | 4.48 — all fit T4 |
| fp32  | 1.20 GiB | 2.16  | 3.13   | 5.07     | 7.01   | 8.94 — all fit T4 |

**qwen3-0.6B (qwen3_ls ckpt):** activations ≈ 70 MB/sample fp16 (~9× granite — full S×S
attention, 28 layers, hidden 1024; no ModernBERT local-window discount)

| dtype | weights  | bs 64           | bs 128     | bs 256            | bs 384      |
| ----- | -------- | --------------- | ---------- | ----------------- | ----------- |
| fp16  | 1.11 GiB | 5.56 ✓          | **9.99 ✓** | 18.87 **OVER T4** | OOM on 24GB |
| fp32  | 2.23 GiB | 10.85 ✓ (tight) | 19.48 OVER | OOM               | —           |

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

| combiner                                  | fitted params | note                                                                                                                                                           |
| ----------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **soft-uniform** (mean of member softmax) | 0             | the DEFAULT (ships unless beaten honestly); prob-space neutralizes LS-vs-CE logit temperature                                                                  |
| logit-mean                                | 0             | scale-sensitive: implicitly overweights sharp (CE) members                                                                                                     |
| geo-mean (product of experts)             | 0             | punishes any member's low prob — stricter consensus                                                                                                            |
| hard-vote (majority)                      | 0             | discards confidence; ties broken by soft-uniform                                                                                                               |
| rank-mean                                 | 0             | scale-free; coarse at C=14                                                                                                                                     |
| weighted prob mean ⚠️                     | M             | fit on slice = calibration-adjacent + E8 mis-rank risk → reported as 2-fold honest CV **and** full-slice upper bound; shipping it needs explicit user sign-off |
| stacking (LR on member probs) ⚠️          | M·C·C         | max capacity, same objections; 2-fold honest CV only                                                                                                           |
| weight-space soup / SWA                   | 0             | NOT logit-level — needs shared-init retrains (E12); optional arm                                                                                               |
| distillation (Path 1B)                    | —             | ensemble → single model; the constraint-dissolving variant                                                                                                     |

## Path-1B distillation — ❌ CLOSED 2026-07-12 (plan user-approved 2026-07-10; ran on vast; verdict below)

**Teacher selection** (slice numbers for the record — but the trio is chosen because it is
**LB-certified**, the only teacher whose test-distribution quality is measured; the slice
mis-ordered ensembles twice today):

| teacher candidate                                      | slice  | verdict                                                                                                                             |
| ------------------------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **T1 = LB-certified trio** is3+aum06+e25c (LB 0.78719) | 0.7850 | ✅ PRIMARY                                                                                                                          |
| 5-union of both LB ensembles (+is2, estack)            | 0.7824 | ✗ slice-worse, no new diversity axis                                                                                                |
| **T2 = trio + champion + qwen3_ls** (6 members)        | 0.7842 | ✅ DIVERSITY BET — the only mechanism that gets qwen3 knowledge into a submittable model; slice structurally under-ranks qwen3 (E8) |

**Student run set** (from-scratch granite champion recipe, bs16×ga1+group_by_length; ~35 min/student on a 3090, runs on vast.ai):

| #   | teacher                              | KD setting                                                                               | tests                                                                 | slice macro-F1 (full_data 3.5k; Δ vs champion anchor 0.7803) |
| --- | ------------------------------------ | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------ |
| 1   | T1 trio                              | vanilla α=0.7, **T=2** (teacher = mean of 3 LS models → already soft; don't over-temper) | the workhorse                                                         | 0.7691 (**−0.0112**)                                         |
| 2   | T1 trio                              | soft-labels-only α=1, T=1                                                                | teacher-as-better-labels (E22: ~6% labels noisy)                      | 0.7635 (**−0.0168**) (worst)                                 |
| 3   | T2 5-member (trio+champion+qwen3_ls) | vanilla α=0.7, T=2                                                                       | cross-backbone transfer (qwen3 → granite)                             | 0.7696 (**−0.0107**)                                         |
| 4   | champion only                        | vanilla α=0.7, T=2                                                                       | born-again CONTROL — KD-regularizer vs ensemble-knowledge attribution | 0.7728 (**−0.0075**)                                         |

**VERDICT (all 4 students done 2026-07-12): path 1B CLOSED — distillation is net-negative under
this harvest scheme.** Champion anchor 0.7803; every student landed 0.007–0.017 below it, and
the ensemble-taught students (0.7691/0.7696) even UNDER the single-teacher control (0.7728).
Per-epoch curves: all four peaked at epoch 2 and flattened/declined at 3 → NOT undertrained;
they converged to a lower plateau.

**Why (mechanism, fits the numbers):** all teachers were harvested on the rows they trained on
(full_data members) → near-one-hot, ≈ the labels incl. the ~6% noise (E22) → students received
almost no dark knowledge AND lost label smoothing (`--distill_from` is not combinable with
`--loss ls`). Plateau ≈ champion − LS gain: 0.7803 − 0.0107 (E9) ≈ 0.7696 ✓. Ranking confirms:
the harder the student leaned on teacher targets (α=1 soft-only worst at 0.7635), the worse.

**Documented escape hatches if 1B is ever revisited (target-side, not schedule-side):**
(1) OOF-harvested teachers — 5-fold members label only their unseen fold, honest uncertainty
survives; (2) keep LS inside the CE term alongside KD (small finetune.py change). **Both queued
2026-07-12 as E30 phase 1** (session-grouped OOF campaign; E29 learned-head consumes the same
harvest); direct trio submission (LB 0.78719) stays the champion path meanwhile.

Run note: students trained on vast.ai 3090s, ~31 min each. Artifacts:
`output/pat/ft_*e26s_*`, `sbatch/logs/train_e26s_*.log`.

Execution: 6 member harvests over 70k (`distill_teacher.py`; per-member caches shared by both
teachers; qwen3_ls resumes its partial cache) ≈2 h on healthy GPUs → build T1/T2 npz →
4 students overnight → slice-read → submit best 1–2. Endgame note: distilled students are
themselves ensemble MEMBERS — a trio of students is the same 6:52/836M shape with stronger parts.

## Path-1B distillation taxonomy (user ask 2026-07-10)

| #   | method                                                                                              | knobs      | code status                                     | in run set?                                                   |
| --- | --------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------- | ------------------------------------------------------------- |
| 1   | vanilla response KD (Hinton): α·T²·KL + (1−α)·CE, teacher = uniform prob mean                       | α, T       | ✅ `--distill_from/alpha/T`                     | ✅ (α .7, T 3) + (α .5, T 2)                                  |
| 2   | soft-labels-only / label refinery (α=1, T=1 — teacher replaces labels)                              | —          | ✅ same path                                    | ✅ ×1 (E22: ~6% labels noisy → teacher may beat labels)       |
| 3   | born-again control: teacher = champion ALONE                                                        | —          | ✅ same path                                    | ✅ ×1 — decomposes KD-regularizer vs ensemble-knowledge       |
| 4   | transductive KD: teacher soft-labels `data/test.jsonl`, student trains on train+test (KD-only rows) | mix ratio  | 🔨 small finetune.py ext                        | ⏸ pending user RULES check (semi-supervised on provided data) |
| 5   | multi-teacher weighted KD                                                                           | member w   | ✅ (weights in teacher build)                   | only if honest-CV combiner read favors weighting              |
| 6   | feature/attention KD (TinyBERT/MiniLM)                                                              | layer maps | ❌ cross-arch + cross-serialization projections | skip                                                          |
| 7   | EnD² (Dirichlet distribution distillation)                                                          | —          | ❌                                              | skip — argmax metric only sees the mean                       |
| 8   | online DML / noisy-student rounds                                                                   | —          | ❌                                              | defer — multiplied cost                                       |

Students: FROM-SCRATCH granite champion recipe (E24 recovery-trap lesson) + optional qwen3-student
arm (1B′, highest ceiling, 9:18 budget). ~1.5 GPU-h per granite student; 4-run set ≈ 6 GPU-h.

## Results — Phase 0 screen (2026-07-10; 26/34 models cached pre-GPU-crash — all heavy hitters in; missing: a24_anchor, e11_tapt, v1-coreset family, coreset_base)

Slice = shared 3.5k held-out; noise ±0.003 → the pair options are statistically tied; E8 lesson
(slice mis-ranks LB, especially cross-backbone) applies to everything below.

**Singles:** champion e8a_ls **0.7801** (✓ replicates 0.7803) — nothing beats it alone.

**Ensembles (soft-uniform):**

| set                         | members                                         | F1              | Δ champ                                                    |
| --------------------------- | ----------------------------------------------- | --------------- | ---------------------------------------------------------- |
| best pair                   | e12soup_is3 + e25c_richmeta                     | **0.7840**      | +0.0039 (cross-serialization: richargs+richmeta)           |
| best richargs-only pair     | e12soup_is2 + e_stack_tapt_ls                   | **0.7832**      | +0.0031 (single-serialization packaging)                   |
| best champion-anchored pair | e8a_ls + e22_aum06                              | 0.7820          | +0.0019                                                    |
| **greedy-4**                | e8a_ls + e22_aum06 + e12soup_is3 + e16b_depth11 | **0.7851**      | **+0.0050**                                                |
| greedy4 + qwen3_ls          | (teacher candidate)                             | 0.7834          | qwen3_ls HURTS on slice (but slice under-ranks qwen3 — E8) |
| greedy4 + e25c / + both     |                                                 | 0.7840 / 0.7845 | no gain                                                    |

**Combiner shoot-out (both member sets): soft-uniform WINS or ties everything.**
Hard-vote ≈ tied (0.7840/0.7848); logit-mean & geo-mean crater on the greedy set (0.7757 —
the predicted LS/CE temperature mismatch); rank-mean catastrophic at C=14 (0.4970); **every
fitted combiner loses its honest 2-fold read** (weighted 0.7821/0.7791, stacking 0.7776/0.7799
vs uniform 0.7840/0.7851) while its full-slice fit "wins" — textbook overfit, validating the
no-fitted-weights invariant. **Shipping combiner = uniform mean of softmax.**

**Notable:** greedy picked e22_aum06 SECOND despite its solo failure (−0.004) — denoised-model
errors decorrelate from the champion's; solo Δ ≠ ensemble value. e16b_depth11 (a pruned model!)
adds +0.001 as 4th. Ensemble diversity ≠ single-model quality — the screen was worth it.

**Interpretation for the two paths:**

- **1A pair:** all pair options tied within noise → choose on engineering risk: richargs-only
  (is2 + e_stack_tapt, 0.7832) ships with the EXISTING script unchanged; the 0.7840 cross-serialization
  pair needs a per-member-serializer script extension for +0.0008 (noise). Both clear the +0.003 gate.
- **1B teacher:** greedy-4 (0.7851) is the slice-optimal teacher; qwen3_ls variant (+1h harvest)
  as an optional second teacher despite the slice read, since the slice provably under-ranks qwen3 on LB.

_(8 pool stragglers were never appended — the trio outcome made them moot)_

## Results — LB (submitted 2026-07-10, parity waived by user; LB itself validated the pipeline)

| zip                                  | slice                         | **LB**                      | Δ prior SOTA 0.77931 | time     |
| ------------------------------------ | ----------------------------- | --------------------------- | -------------------- | -------- |
| **E26-A** is2+estack (richargs pair) | 0.7832                        | **0.78548** 🥇 **NEW SOTA** | **+0.0062**          | **4:30** |
| E26-B is3+e25c (cross-serialization) | 0.7840                        | 0.78498                     | +0.0057              | 4:46     |
| s43 leak probe (single, fp32)        | 0.7955 leaked / 0.7758 honest | 0.77427                     | −0.0050              | 5:12     |
| **E26-C** is3+aum06+e25c trio        | 0.7850                        | **0.78719** 🥇 **NEW SOTA** | **+0.0079**          | **6:52** |

Readings: (1) **ensembling is a step-change**: +0.0081 over the granite champion single — the
biggest jump since LS; and the slice UNDER-predicted LB (0.7832 → 0.78548) while mis-ordering
the two pairs (preferred B; LB says A) — E8's slice-mis-rank cuts both ways. (2) **Speed thesis
confirmed and exceeded: the fp16 bs-256 pair (4:30) is FASTER than the single fp32 bs-64
champion (5:06)** — a third member fits both caps (≈840M, ~6:45). (3) **s43 leak confirmed on
LB** (0.77427 < champion 0.77738, ≈ its honest 0.7758): the 0.7955 was memorization; also a
same-recipe seed reroll ≠ champion (seed luck real). (4) The LB scores validate the whole
pruned+fp16+dual-serialization pipeline end-to-end (parity gate retroactively moot for these
two zips; keep it for future builds). (5) **Trio (07-10 evening): 3rd member scaling CONFIRMED —
0.78719 (+0.0017 over pair)**; slice under-predicted again (+0.0022); marginal member cost ≈2:22
→ a 4th member (~9:14, ~1.1GB) breaks BOTH caps at safe prune margins → trio ≈ the direct-package
ceiling; further gains go through distillation (path 1B) or smaller members.

## Reproduce

- `experiments/ensemble/screen_ensemble.py` — Phase-0 harvester + greedy selection (logits cached
  `analysis/cache/e26_screen_logits.npz`, idempotent; `--limit` smoke).
- Phase-1 packaging: reuse the granite zip build (`submission-build-recipe` memory,
  `submit_0707_granite_ls_fp16.zip` as the fp16 reference) + `src/prune_vocab.py` for the
  granite-richargs prune; parity gate vs each member's own val logits before zipping.
