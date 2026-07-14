# Compression program — index

Model-compression work, **one file per method family**. This file is the index +
cross-cutting protocol; per-method detail lives in the linked files.

**Two model tracks, one shared method taxonomy:**
- **granite** (`ibm-granite/granite-embedding-311m-multilingual-r2`, ModernBERT: 22 layers,
  H=768, GeGLU I=1152, vocab 262,152) — the deployed primary. Combination search = **E31**.
- **qwen3** (`Qwen/Qwen3-Embedding-0.6B`, decoder-style: 28 layers, FFN 44%, vocab 151k) —
  higher compression headroom (mass is in the compute stack). Combination search = **E45**.

Per-model results transfer across backbones only as *evidence about the method*, never as a
granted result — always re-measure (E16 free on qwen3 vs E16b VOID on granite lesson).

**Two objectives, tracked per method:** **Size ↓** (zip ≤1GB; embeddings dominate granite) ·
**Speed ↑** (10-min/30k budget; 본선 speed score 10%; compute stack dominates).

---

## Measurement protocol — report ALL THREE (user 2026-07-14, standard for every method)

When measuring any compression method's effect on model performance, always report all three,
**vs the uncompressed original on the SAME fixed eval slice, uncalibrated logits** (project
rule [[no-logit-calibration]]). They are ordered from coarsest to finest — a method can be
flat on (1) yet clearly moving on (2)/(3):

1. **ΔF1** — macro-F1(compressed) − macro-F1(original). The headline (competition metric).
2. **Prediction drift** — fraction of samples where `argmax` differs from the original
   model's prediction (report flip-rate, or agreement % = 1 − flip). Catches behavioral
   change the aggregate F1 hides: F1 can stay flat while many predictions swap via
   compensating errors.
3. **Logit change** — distance between compressed and original logits: mean & max `|Δlogit|`,
   plus mean softmax-KL(original ‖ compressed). The finest signal; moves before predictions
   flip. Good early-warning that a knee is near.

Rationale: (1) is what the LB scores, but it's noisy and can mask drift; (2) measures how much
the model's *behavior* changed regardless of correctness; (3) is the most sensitive and
predicts where the accuracy cliff is before it shows up in (1). ⚠️ Retrofit: the E31 A1 depth
probe reported only (1) — re-run to add (2)/(3) when convenient.

### Which data — full_data model vs the 3.5k held-out (user 2026-07-14, MUST distinguish)

The models we compress (t031 etc.) are **`--full_data`** models. That flag
([finetune.py](../../src/finetune.py) ~L1236): `split_indices(test_size=0.2, seed=42)` →
14k val, then **75% of that val is folded into training** (→ ~69k trained) and only the
remaining **25% = the 3.5k slice (`va_eval`) is held out** — the model never saw it. Repro:
```python
tr, va = split_indices(y, test_size=0.2, seed=42)                       # 14k val
va_train, va_eval = train_test_split(va, test_size=0.25,
                                     stratify=y[va], random_state=42)    # va_eval = 3.5k held-out
```
- **Measure compression on the 3.5k held-out (`va_eval`) — the ONLY honest eval for a
  full_data model.** Report it as "3.5k held-out".
- **Do NOT** measure on a session-grouped fold or any other subset of the 69k — those are
  **in-sample** for a full_data model → inflated. (The earlier E31-A1 depth probe used a
  session fold → its 0.787/0.803 baselines are IN-SAMPLE, not the 3.5k held-out; the LB is
  0.793. The pruning *deltas* are still roughly valid — both variants saw the same rows — but
  the absolute F1 is not the model's true accuracy. **Retrofit the depth probe onto `va_eval`.**)
- `--all_data` (E28 final-submission mode) folds *everything* in with **no held-out** → no
  honest local eval exists at all; only the LB judges those.
- Always label a measurement **"3.5k held-out"** vs **"in-sample"** so the two are never
  conflated.

### Recovery-training uplift — report zero-shot AND post-recovery (user 2026-07-14)

Structured compression = prune/factor **then recovery fine-tune**. Recovery is half the
method — always report both stages and how much recovery closed, so its contribution is
explicit (never report only the zero-shot damage or only the final number):

1. **Zero-shot** — compressed model, NO retraining (raw damage) — the 3 metrics vs baseline.
2. **Post-recovery** — after recovery-FT — the 3 metrics vs baseline.
3. **Recovery uplift** = post-recovery − zero-shot (how much recovery-FT recovered).
4. **Net vs baseline** = post-recovery − uncompressed baseline — the shippable verdict;
   Stage-A gate: net ΔF1 ≥ −0.002.

- Recovery-FT trains on the **same full_data training rows** (never the 3.5k held-out); eval
  on the **3.5k held-out**. Sweep recovery epochs/LR — the uplift number says whether more
  would help (big zero-shot drop fully closed = GO regime; small drop recovery can't close =
  NO regime; e.g. qwen3 depth-14 recovered a large zero-shot loss to −0.0005 = GO).
- Also worth the **from-scratch anchor** (E24 warm-start trap): does recovery-from-pruned-init
  beat training the compressed architecture from scratch? Report if measured.

---

## Work division (user 2026-07-14): every method × BOTH models, separately

For **each** compression method, run the experiment on **granite AND qwen3 separately** and
measure the impact on each model — two parallel per-method tracks, each scored vs its own
baseline below (3-metric protocol + recovery uplift). Never assume cross-backbone transfer
(and per-checkpoint transfer is also false: a BI prune-order derived from t031 cost −0.10 F1
at keep-18 on a *sibling* AWP granite vs −0.04 with that model's own order — layer redundancy
is CHECKPOINT-specific, always re-probe the model you actually prune).

## Compression baselines (measure every method against these)

| Track | Baseline zip | Model | LB | vocab | layers | dtype | size | 30k time |
|---|---|---|---|---|---|---|---|---|
| **qwen3 (E45)** | `submit_0707_qwen3_ls.zip` | qwen3-0.6b E8b+LS, richargs | **0.77921** 🥇 | 29,657 (pruned) | 28 (full) | fp16 | 830M | 9:18 (over T4 budget) |
| **granite (E31)** | `submit_0714_awp_t031.zip` | granite-311m E8a+LS+AWP (t031) | 0.79300 | 262,152 (full) | 22 (full) | fp16 | 659M | ~5:06 |

- **qwen3 baseline chosen (user 2026-07-14):** `submit_0707_qwen3_ls` — the strongest qwen3
  single (the accuracy edge E45 must "hold ≥ 0.77921"), full 28 layers so structural
  compression is a clean delta, and already vocab-pruned + fp16 (the always-on base). The
  `depth14`/`E4z` zips below are *compression results* to compare against it, not baselines.
- Both are **`--full_data`** models → honest eval = the **3.5k held-out** (see protocol above);
  report all 3 metrics vs the baseline.
- Existing qwen3 compression points to compare: `submit_0708_qwen3_depth14.zip` (E16z, depth
  28→14) · `output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e4_ffn_r512_recover_e8b` (E4z FFN-factor
  r512, +0.0045). Note: qwen3 baseline is **9:18/30k — over the T4 budget**, which is exactly
  why compression is the *enabler* here, not just an optimization.

## Method-family index

| # | Family | File | Status (granite unless noted) | Size↓ | Speed↑ |
|---|---|---|---|---|---|
| 1 | Quantization (fp16 / int8) | [quantization.md](quantization.md) | fp16 ✅ IN USE · int8/TensorRT 🏃 separate session | ✅ | ✅ |
| 2 | Vocab pruning | [vocab-pruning.md](vocab-pruning.md) | ✅ IN USE — biggest size lever (emb 64.6%) | ✅✅ | — |
| 3 | Depth pruning | [depth-pruning.md](depth-pruning.md) | ✅ DONE both: granite best keep-18 net **−0.0038** (~18% layers), qwen3 best keep-14 net **−0.002..−0.004** (≈E16, ~free — NOT a gain). Neither clears −0.002 gate cleanly; recovery hurts mild cuts; ⚠️ ModernBERT reload trap fixed | ✅ sm | ✅ modest |
| 4 | Width pruning | [width-pruning.md](width-pruning.md) | 🔲 unexplored — box-safe probes planned | ✅ sm | ✅ |
| 5 | Low-rank factorization | [low-rank-factorization.md](low-rank-factorization.md) | 🔲 granite (low yield); qwen3 ✅ +0.0045 (E4) | ✅ sm | ~ marg |
| 6 | Knowledge distillation | [knowledge-distillation.md](knowledge-distillation.md) | ❌ CLOSED — trio≠distillable (E26-1B/E30) | ✅✅✅ | ✅✅✅ |
| 7 | Token reduction (LTP) | [token-pruning.md](token-pruning.md) | 🕐 DEFERRED — awaits re-spec | — | ✅✅ |
| 8 | Sparse / efficient attention | [sparse-attention.md](sparse-attention.md) | 🔲 low yield — granite already native-sparse | — | ~ sm |
| 9 | Unstructured pruning | [unstructured-pruning.md](unstructured-pruning.md) | ❌ DEPRIORITIZED — no HW kernels (2:4 only) | (storage) | ❌ |

Inference engineering (E17 token-budget batching) is free speed, ships in every zip — not
compression, tracked in EXPERIMENTS.md §E17. MoE = E29 arm ③ (accuracy play, not compression).

**Structured vs unstructured** (primer): structured = remove whole units (layers, heads,
channels, tokens, vocab) → smaller *dense* model, unconditional speedup+size, pair with
recovery FT. Unstructured = zero individual weights → no GPU speedup without sparse kernels.
**We do structured only** (user 2026-07-12). Detail: [unstructured-pruning.md](unstructured-pruning.md).

---

## Combination SEARCH — E31 (granite) / E45 (qwen3)

Structured methods only; with fleet GPUs we SEARCH the combination space, not a fixed ladder.
Full specs: EXPERIMENTS.md §E31 / §E45.

- **Base (always-on, packaging-time):** vocab-prune + fp16.
- **Stage A — parallel axis screens** vs a shared from-scratch anchor (E24 warm-start trap):
  depth · width · FFN low-rank · (token pruning parked). Training-free probes first (BI,
  zero-shot drop, SVD-degradation), recovery-FT only for survivors. Per-axis gate: ΔF1 ≥
  −0.002 AND ≥1.2× measured speedup (or real params cut) — reporting all 3 metrics above.
- **Stage B — factorial over survivors:** compose all prunes structurally FIRST, then **one
  joint recovery FT per combo** (recovery must see the final architecture). Pareto pick on
  (macro-F1, ms/sample→30k wall-clock, params).
- **Stage C — transfer:** winning combo → the post-E30/E29 optimal model → package (parity
  gate) → one LB slot. LB judges (slice mis-ranks — E8/E26).

**qwen3 (E45)** is the higher-headroom mirror: mass IS the compute stack (FFN 44% / attn 30%
/ emb 26%), and both axes are already proven individually — FFN-factor r512 +0.0045 (E4) and
depth 28→14 ~free (E16) — but never combined. Bet = the depth×FFN×attn product. Motivation:
qwen3 had the accuracy edge (LB 0.77921 > granite 0.77738 @E8) but 9:18/30k = un-shippable;
compression is the enabler. Detail per axis in the linked method files.

---

## Stage-B combination results (granite t031, honest 3.5k held-out, 2026-07-14)

| combo | macro-F1 | net ΔF1 | flip | KL | vs single axes |
|---|---|---|---|---|---|
| **depth-18 × FFN-width-75%** (joint recovery) | **0.78407** | **−0.0017** ✅ | 5.4% | 0.020 | BEATS depth-18 alone (−0.0038) |

**Headline: the combination CLEARS the −0.002 gate** (first granite structured combo to do so),
at ~18% fewer layers AND 25% fewer FFN neurons — and stacking width onto depth with ONE joint
recovery FT beat depth-alone (−0.0017 vs −0.0038). Interpretation: extra structured pruning +
joint recovery acts as regularization (or ≥noise-level improvement). Stacks on the nf4 quant
(−30%, [[E41 qwen3 AWP ceiling]] memo) → a meaningfully faster granite at ~baseline accuracy.
Standalone width arms (keep-75/50/25) + FFN low-rank (probe: gentle, small yield since FFN is
18.7%) pending/parked — see [depth-pruning.md](depth-pruning.md) / [width-pruning.md](width-pruning.md).

## Open questions

- granite depth recovery-FT: does keep-18's −0.015 close to net-≈0? (real GPU box).
- Retrofit E31 A1 (and all probes) to report the 3-metric protocol, not just ΔF1.
- Retained-vocab fraction + exact MB saved for granite vocab-prune (record at next build).
- Does the DACON T4 stack dispatch 2:4 sparse kernels? (gates unstructured).
- Wall-clock profile of granite on the T4: attention vs FFN vs embedding share (decides
  whether sparse-attention / FFN work has any speed payoff).
