# E46 — Quantization (two arms: granite + qwen3)

**Structure (per user, 2026-07-14 — shared with the main compression session):** every quantization
method is run on **both backbones separately**, measuring impact per model. **Granite arm first**, qwen3 arm after.

| arm | baseline model | source zip | status |
|-----|----------------|-----------|--------|
| **A. granite** | `granite-311m-e8a-ls-awp-t031` (E38 AWP single, LB **0.79300**) | `submit_0714_awp_t031.zip` ← **canonical granite baseline for E46** | ⏳ running |
| **B. qwen3** | qwen3 champion (LB 0.7682-class, pruned-vocab weights) | `submit_0703_qwen3_pruned.zip` (per [[qwen3-champion-weights-loading]]: pruned embed + remap + custom last-non-pad pooling) — *confirm before starting arm B* | ⬜ queued after granite |

---

## Arm A — granite (`granite-311m-e8a-ls-awp-t031`)

**Model under test:** `submit_0714_awp_t031.zip` → `granite-311m-e8a-ls-awp-t031`
(ModernBertForSequenceClassification, 22 layers, hidden 768, 12 heads, vocab 262152, 14 tool-labels; stored fp16, 624 MB safetensors)

**Provenance:** full_data recipe — ~66.5k rows train, 3.5k rows val (of 70k total). LB reference (fp16) = **0.79300** (E38 optuna-tuned AWP single, clears #12). See [[e38-awp-optuna-sota]].

**Goal:** quantize weights (and where supported, activations) to lower precision, measure (a) how much predictions drift from the fp16 reference across **all** data, and (b) accuracy where labels are trustworthy. Secondary: size + latency/throughput.

**Hardware (test box):** ArchServer, RTX 4060 (Ada, SM 8.9), 8 GB. ⚠️ Ada supports FP8 natively; a **T4 deployment target would NOT** (no FP8, different INT8 kernels) — treat FP8 numbers as headroom, not portable.

---

## Measurement protocol — the 3 dimensions to always report

> **Canonical spec (per user, 2026-07-14 — keep for all future quantization/compression runs).**
> When measuring a quantization/compression method's effect on model performance, always report
> these **three** dimensions vs the un-quantized fp16 reference, over all 70k rows AND the 3.5k val slice:

| # | dimension | metric(s) | why |
|---|-----------|-----------|-----|
| **1** | **change in F1** | macro-F1(quant) and **ΔF1 = F1(quant) − F1(fp16 ref)** | task metric; the number that decides shippability |
| **2** | **change in prediction** | **flip% = fraction of rows whose argmax ≠ fp16 ref argmax** | how much the deployed decisions actually move (label-free) |
| **3** | **change in logit** | **L1, RMSE, max\|Δ\|, cosine** of (quant_logits − ref_logits) | raw signal distortion upstream of argmax (label-free; no calibration per [[no-logit-calibration]]) |

Supporting: `acc_all` / `acc_val` (accuracy), `prob_KL`, on-disk `size_MB`, `infer_s` (throughput). F1 uses macro over the 14 classes (competition metric). Dimensions 2 & 3 are label-free; dimension 1 needs labels.

- **Reference:** shipped fp16 PyTorch model (`torch_dtype=float16` + cuda autocast — exactly `script.py`), raw argmax over 14 classes. acc_all=0.79970, acc_val=0.77829.
- **Data:** all **70k** labeled rows from `data/train.jsonl` (the "all data"; `test.jsonl` is a 5-row stub). richargs serialization = `src.data.serialize(rich_meta=True, arg_basenames=True)`, byte-identical to training.
- **Val slice (3.5k):** exact t031 full_data holdout — seed=42, stage-1 stratified 80/20 then stage-2 75/25 of the 20% (net 5%). ⚠️ **row-stratified → session-leaky, AND used for t031 checkpoint selection → acc_val/f1_val are optimistic.** Trust the *drift* (dims 2–3) and ΔF1, not the absolute val number.
- **Aggregation:** `aggregate.py` recomputes all 3 dimensions uniformly from each method's saved logits, so every method (native or TRT) is scored identically.

**Location:** this experiment lives under `experiments/compression/quantization/` (moved into the model-compression track 2026-07-14). See sibling `../quantization.md` (compression-session method note) and `../results.md` (compression board).

**DACON deployment target (decides which methods matter):** T4 GPU 16 GB, 3 vCPU, 12 GB RAM, offline, **inference ≤ 10 min**, package install ≤ 10 min, submission ≤ 1 GB. → int8 (TRT/bnb) is the accelerated+portable path; FP8 and torchao-int4 do **not** run/accelerate on T4 (measured here as headroom only).

---

## Methods to test

### TensorRT track (ONNX export → TRT engine; Ada precisions)
| # | method | precision | activations | calibration | notes |
|---|--------|-----------|-------------|-------------|-------|
| T0 | TRT FP32 | fp32 | fp32 | — | functional reference vs PyTorch (export sanity) |
| T1 | TRT FP16 | fp16 | fp16 | — | expected ~lossless; throughput baseline |
| T2 | TRT BF16 | bf16 | bf16 | — | model native dtype is bf16; compare vs fp16 |
| T3 | TRT INT8 (entropy) | int8 | int8 | entropy (KL) | W8A8 PTQ, ~512-row calib set |
| T4 | TRT INT8 (minmax) | int8 | int8 | minmax | W8A8 PTQ, compare calibrators |
| T5 | TRT FP8 (E4M3) | fp8 | fp8 | Ada only | W8A8-fp8; **not portable to T4** |

> INT4 is **N/A** on this track: standard TensorRT has no INT4 for encoders (INT4 is TRT-LLM-only, autoregressive). Covered instead in the PyTorch track as weight-only storage.

### PyTorch-native track (no ONNX; direct drift study; more portable)
| # | method | precision | scheme | lib | notes |
|---|--------|-----------|--------|-----|-------|
| P0 | BF16 plain | bf16 | no quant | torch | isolates bf16-vs-fp16 dtype delta (torchao arm runs in bf16, see caveat) |
| P1 | INT8 weight-only | W8A16 | per-channel | torchao | cheapest, usually near-lossless |
| P2 | INT8 dynamic | W8A8 | dynamic act | torchao | closer to TRT INT8 behavior |
| P3 | ~~INT4 weight-only~~ | W4A16 | tinygemm g128 | torchao | **DROPPED** — needs `mslk` kernel lib (unavailable); non-portable to T4 anyway; 4-bit covered by P6/P7 |
| P4 | FP8 dynamic | W8A8-fp8 | rowwise | torchao | Ada float8; compare vs TRT FP8 |
| P5 | LLM.int8() | W8A8 | outlier-fp16 mix | bitsandbytes | outlier-preserving |
| P6 | NF4 | 4-bit weight | nf4 g64 | bitsandbytes | non-uniform 4-bit grid |
| P7 | FP4 | 4-bit weight | fp4 g64 | bitsandbytes | uniform 4-bit float |

**torchao caveat (affects P1/P2/P4 interpretation):** torch 2.7 < torchao's compiled-kernel floor (2.11) → quantized matmuls take the *reference* path. In fp16 that path **overflows to NaN** on real inputs; in bf16 it is clean (probe-verified: bf16+int8wo matches plain fp16 logits to ~0.02). So the torchao arm runs **bf16 weights/compute**: its drift vs the fp16 REF includes a small dtype component — subtract the P0 (bf16 plain) row to isolate pure quant error. torchao **latency is meaningless** here (reference path); bnb latency is real (compiled kernels).

### Degradation-mitigation add-ons (apply to whichever base loses most)
- **SmoothQuant** activation-migration scale (helps INT8 W8A8) — apply before T3/P2.
- **Percentile/MSE clipping** instead of minmax for INT8 calibration.
- **Keep-sensitive-in-fp16:** embeddings + classifier head + final LayerNorm left high-precision.
- **Group-size sweep** for 4-bit (g128 → g64 → g32).

### Recovery-training agenda → **E47** (queued in EXPERIMENTS.md; 🚫 NOT on local ArchServer — ocean/vast only)
Planned regardless of E46 outcome (user 2026-07-14), gated per-scheme on E46 showing real ΔF1 loss:
1. **QLoRA-style adapter recovery** — frozen bnb-NF4/int8 base + fp16 LoRA on top (cheapest; no backprop into quantized weights — that's impossible by construction).
2. **Post-quant KD** — fp16 champion as teacher, quantized model as student, short distillation fine-tune.
3. **QAT-int8 → TRT explicit quant** — fake-quant fine-tune (STE) → Q/DQ ONNX → TRT; heaviest, only if TRT-int8 PTQ calibration tricks (calibrator choice, SmoothQuant, clipping) can't hold accuracy.

⚠️ All three are warm-start fine-tunes of the champion → the **E24 recovery trap** applies ([[token-selection-null-recovery-trap]]: champion+3ep@2e-5 → ~0.763 regardless of change). Use careful LR/epochs and from-scratch-vs-from-scratch anchors. TRT engines themselves are frozen — recovery always happens in PyTorch *before* export.

---

## Results

_Generated by `aggregate.py` — all 3 dimensions from saved logits, **train (66.5k in-sample) vs val (3.5k held-out) split**. Updated as methods land._

**PyTorch-native track** (measured on RTX 4060; torchao latency N/A — compiled kernels skipped, torch<2.11 → reference path; drift/F1 valid):

| method | scheme | T4 | size MB | **F1 train** | **ΔF1 train** | **F1 val** | **ΔF1 val** | **flip% train** | **flip% val** | **logit L1** | **logit RMSE** | **max\|Δ\|** | **cos** | acc train | acc val | infer s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 (REF) | fp16 shipped | yes | 622 | 0.8091 | +0.0000 | 0.7858 | +0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 1.00000 | 0.8008 | 0.7783 | — |
| fp32 | fp32 (no quant) | yes | 1244 | 0.8091 | +0.0000 | 0.7858 | +0.0000 | 0.009 | 0.000 | 0.0002 | 0.0004 | 0.001 | 1.00000 | 0.8008 | 0.7783 | 1359 |
| bf16 | bf16 (no quant, dtype baseline) | yes* | 622 | 0.8094 | +0.0003 | 0.7847 | −0.0010 | 0.173 | 0.143 | 0.0028 | 0.0038 | 0.008 | 1.00000 | 0.8010 | 0.7777 | 558 |
| int8_wo | torchao int8 W8A16 (bf16) | yes | 311 | 0.8094 | +0.0003 | 0.7851 | −0.0007 | 0.221 | 0.143 | 0.0065 | 0.0081 | 0.016 | 0.99998 | 0.8012 | 0.7777 | 603* |
| int8_dyn | torchao int8 W8A8 (bf16) | yes | 311 | 0.8091 | +0.0000 | 0.7849 | −0.0009 | 0.630 | 0.571 | 0.0231 | 0.0291 | 0.059 | 0.99976 | 0.8007 | 0.7786 | 830* |
| bnb_int8 | bnb LLM.int8() W8A8 | yes | 311 | 0.8092 | +0.0001 | 0.7858 | **+0.0000** | 0.072 | 0.057 | 0.0037 | 0.0050 | 0.009 | 0.99999 | 0.8009 | 0.7783 | 1165 |

**PAUSED 2026-07-14 (user needs the local GPU):** remaining — bnb_nf4 → bnb_fp4 → fp8_dyn, then TRT fp32-sanity/fp16/int8×2 (ONNX exported, parity 1e-6, TRT 24.12 image pulled). Resume: `run_all_native.py bnb_nf4 bnb_fp4 fp8_dyn`. `*` torchao latency = reference path, not meaningful; bnb latency is real kernels (LLM.int8() ~2× slower than fp16 on this 4060 — outlier-decomposition overhead, expected).

---

## Findings — Arm A (granite)

_(written after runs — design/results/methodological caveats only, per [[results-md-method-report]])_

---

## Arm B — qwen3 (queued, starts after Arm A completes)

⚠️ **Do NOT run arm B on ArchServer (local 4060/8GB)** — user directive 2026-07-14: qwen3 inference
may crash the box/session. Run it on ocean/vast (or wherever the user assigns) when arm A is done.

**Baseline (to confirm):** `submit_0703_qwen3_pruned.zip` — the qwen3 champion weights (pruned
embeddings + remap; head pools last non-pad token, NOT attention_mask; assert `h @ scoreᵀ == logits`).
Loading is nonstandard — see [[qwen3-champion-weights-loading]] before building the arm-B reference.

Same protocol as Arm A: fp16 reference over all 70k → per-method ΔF1 / flip% / logit-drift,
train (66.5k in-sample) vs val (3.5k held-out) split, same method list (native + TRT).
Artifacts will use `_work/qwen/` to keep arms separated.
