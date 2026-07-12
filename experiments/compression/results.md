# Branch: compression — granite compression program (status board)

**Primary model = granite** (`ibm-granite/granite-embedding-311m-multilingual-r2`,
ModernBERT-class: 22 layers, H=768, GeGLU I=1152, vocab 262,152). All prior qwen3
compression results are *evidence about the method*, not about granite — transfer must be
re-measured (E16/E16b lesson).

**Two objectives, tracked separately per method:**
1. **Size ↓** — model/package size (submission caps; trio is 836M vs cap, 4th member breaks it).
2. **Speed ↑** — computation / inference wall-clock (10-min/30k budget; trio 6:52; 본선 speed score 10%).

**Sequencing (user 2026-07-12):** first find the optimal model (E30 → E29 line), THEN apply
the validated compression stack to it — compression is the novelty/competitiveness layer,
and it also buys headroom (a compressed member could fit a 4th ensemble slot or TTA).

## Where granite's mass and compute are

| Component | Params | Share | FLOPs share | Implication |
|---|---|---|---|---|
| Embeddings (262k × 768) | 201.3M | **64.6%** | ~0% (lookup) | **THE size lever** — vocab-prune; irrelevant to speed |
| FFN (22 × GeGLU 768↔1152) | 58.4M | 18.7% | ~large | thin already (I=1.5H); limited low-rank room |
| Attention (22 × 4·H²) | 51.9M | 16.7% | ~large | already sparse natively (local-128 window, global every 3rd layer) |

Contrast qwen3-0.6B (FFN 44% / emb 26% / attn 30%): the E3/E4 "FFN is the prize" logic
does **not** carry over — on granite the encoder stack is only ~35% of params, so
size gains beyond vocab-prune+fp16 are capped; **stack methods matter for speed, not size**.

## Method-family status board

| # | Family | Method(s) | Status on granite | Size ↓ | Speed ↑ | Evidence / next action |
|---|---|---|---|---|---|---|
| 1 | Quantization | fp16 | ✅ **IN USE** — E26 members packaged fp16, bs256, LB-verified (trio 0.78719) | ✅ ~2× | ✅ | CLOSED per user 2026-07-12 — no int8/int4 work planned |
| 2 | Pruning · vocab | prune_vocab + remap + parity gate | ✅ **IN USE** — part of submission recipe; biggest single lever (64.6% of params) | ✅✅ | — (lookup) | Reusable across same-backbone subs ([[reuse-pruned-tokenizer]]); record retained-vocab fraction for granite when next built |
| 3 | Pruning · depth | ShortGPT/BI layer drop + recovery FT | ⚠️ **UNTESTED — E16b was VOID (bug)**: `--keep_layer_idx` vs `--keep_layers` trained a 1-layer model (0.16/0.24 ≈ random). "Granite can't depth-prune" is NOT a finding | ✅ (≤~35% max) | ✅✅ (~linear in layers kept) | Rerun with `--keep_layers` (fix HELD in repo); FIRST run the BI cosine-redundancy probe on granite to measure per-layer redundancy before committing (qwen3 28→14 was free, E16 — no transfer assumed) |
| 4 | Pruning · structured width | head prune / FFN-neuron slice (Wanda-sp, FLAP) | 🔲 unexplored | ✅ (small mass) | ✅ (real GEMM shrink) | Candidate after depth verdict; dense-shape output → real speedup on any HW |
| 5 | Pruning · unstructured | magnitude / SparseGPT / Wanda sparsity | 🔲 unexplored | (✅ storage only) | ❌ no GPU speedup without 2:4 sparse kernels | LOW priority — masks don't shrink GEMMs; only 2:4 semi-structured (Ampere+) executes faster, unknown if DACON HW/stack uses it |
| 6 | Low-rank factorization | whitened-SVD factor + recovery (SVD-LLM style, `--factor_ffn`) | 🔲 not tried on granite. E4 (qwen3) was net-POSITIVE on accuracy (+0.0045) but only ~15% params, ms/sample never measured — "little acceleration" is correct: it was a size/regularization win | ✅ small (FFN = 18.7%) | ~ marginal | Expected yield LOW on granite: FFN already thin (768↔1152), attn is 4 small H² mats; run the E3-style probe only if stack-compression is needed after depth verdict |
| 7 | Knowledge distillation | trio-teacher → single granite student | 🏃 **ACTIVE** — E26-1B ❌ (in-sample teachers ≈ noisy labels, students 0.7635–0.7728 < 0.7803); **E30 OOF-teacher retry RUNNING** (2026-07-12) | ✅✅✅ (3 models → 1) | ✅✅✅ (~3× vs trio) | THE path to collapse the ensemble; gate: student ≥ champion 0.7803. See [ensemble/results_e30.md](../ensemble/results_e30.md) |
| 8 | Token reduction | LTP learned token pruning (E18, `src/ltp_*` parked) / ToMe fallback | 🕐 DEFERRED (user 2026-07-08) | — | ✅✅ (~2× lit. claim) | Pure speed lever, orthogonal — compounds with depth prune and E17; revive if speed score binds after the stack is chosen |
| 9 | Sparse / efficient attention | window shrink / more local layers | 🔲 unexplored — **but granite is natively sparse already** (ModernBERT: local-128 sliding window, global attn every 3rd layer only) | — | ~ small headroom | Low expected yield; only knobs = shrink `local_attention` / thin global layers + recovery. Park unless profiling shows attention dominates |
| 10 | MoE / gating | E29 arm ③ per-row gating head over trio members (mixture-of-experts over members) | ⛔ GATED — needs E30 phase-0 OOF + calibration ruling; strict gate order ①→②→③ | ❌ (adds KBs) | — | Accuracy play, not compression; recorded here because it's the project's MoE instance. See EXPERIMENTS.md §E29 |
| 11 | Inference engineering | E17 token-budget batching + length-sort | 🟢 READY, pure local work | — | ✅ | Not compression but free speed; goes into every zip; measure on 3090 |

## Method notes — unstructured vs structured pruning (primer)

- **Unstructured**: zero individual weights by importance (magnitude; Wanda = |w|·‖x‖;
  SparseGPT = Hessian-aware one-shot). Reaches high sparsity (50–60%+) at low accuracy
  cost, but the weight matrices keep their shape — **no speedup and no memory saving on
  GPU** unless stored sparse and executed with sparse kernels. The only HW-executable
  variant is **2:4 semi-structured** (2 of every 4 weights zero; Ampere sparse tensor
  cores ≈ up to 2× GEMM). Verdict for us: pursue only as 2:4, and only if DACON's
  inference stack actually dispatches sparse kernels — otherwise it's a paper number.
- **Structured**: remove whole units — attention heads, FFN neurons/channels (Wanda-sp,
  FLAP), layers (= our depth prune), or token positions (= LTP). Output is a **smaller
  dense model**: speedup and size are unconditional, no special kernels. Costs more
  accuracy per parameter removed than unstructured, so it's always paired with recovery
  FT (our E4/E16 regime). Everything that has worked for us (depth, vocab, low-rank
  factoring) is structured; that should remain the default.

## Combination SEARCH — E31 (user 2026-07-12: fleet-parallel search, not a fixed ladder)

**Decision 2026-07-12: structured methods only** (unstructured excluded — no HW kernel
support → no realized benefit). With abundant GPUs, we SEARCH the combination space in
parallel instead of committing to one ordered stack. Full spec: EXPERIMENTS.md §E31.

- **Always-on base** (packaging-time, no training): vocab-prune + fp16.
- **Stage A — parallel axis screens** on the champion recipe (shared anchor 0.7803):
  A1 depth (BI probe → keep-16/keep-11 + recovery; = the honest E16b rerun) ·
  A2 width (FLAP-style head+FFN-channel slice @75%/50% + recovery) ·
  A3 FFN low-rank (granite SVD probe first; factor only if a knee shows) ·
  A4 token pruning (LTP — ⛔ parked until user re-spec; slot reserved).
  Per-axis gate: F1 Δ ≥ −0.002 AND ≥1.2× measured speedup (or real params cut).
- **Stage B — factorial over survivors:** compose all prunes first, then **one joint
  recovery FT per combo** (recovery must see the final architecture; no sequential
  per-axis recoveries). ≤8 combos for ≤3 survivors, fleet-parallel. Read-out =
  interaction table + Pareto pick on (macro-F1, ms/sample, params).
- **Stage C — transfer:** winning combo → the post-E30/E29 optimal model → package
  (parity gate) → one LB slot. LB judges (slice mis-ranks ensembles/compressed models).
- **Every arm reports:** uncal macro-F1 vs the SAME from-scratch anchor (E24 warm-start
  trap) · params (M) · zip size · ms/sample + projected 30k wall-clock (3090 proxy).
- **Orthogonal, ships regardless:** E17 token-budget batching. **Interacts:** if the E30
  KD student gates (≥0.7803), Stage C's target may be the student — the combo transfers.
- Cost: recovery FT ≈ 4–5 h/3090; Stage A ≈ 5–7 trainings + 2 free probes; Stage B ≤ 8
  trainings → ~2 fleet waves.

## Open questions

- **Why did depth pruning "fail" on granite?** It didn't — E16b was an implementation bug
  (1-layer model). The real question is open: how redundant are granite's 22 layers?
  → BI probe + `--keep_layers` rerun.
- Retained-vocab fraction + exact MB saved for granite vocab-prune (record at next build).
- Does DACON inference HW execute 2:4 sparse kernels? (Determines whether unstructured
  pruning is worth anything.)
- ms/sample profile of granite on 3090: attention vs FFN vs embedding share of wall-clock
  (decides whether sparse-attention or FFN work has any speed payoff).
