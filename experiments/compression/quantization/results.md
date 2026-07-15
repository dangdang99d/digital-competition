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
| bf16 | bf16 (no quant, dtype baseline) | ⛔no-bf16 | 622 | 0.8094 | +0.0003 | 0.7847 | −0.0010 | 0.173 | 0.143 | 0.0028 | 0.0038 | 0.008 | 1.00000 | 0.8010 | 0.7777 | 558 |
| int8_wo | torchao int8 W8A16 (bf16) | yes | 311 | 0.8094 | +0.0003 | 0.7851 | −0.0007 | 0.221 | 0.143 | 0.0065 | 0.0081 | 0.016 | 0.99998 | 0.8012 | 0.7777 | 603* |
| int8_dyn | torchao int8 W8A8 (bf16) | yes | 311 | 0.8091 | +0.0000 | 0.7849 | −0.0009 | 0.630 | 0.571 | 0.0231 | 0.0291 | 0.059 | 0.99976 | 0.8007 | 0.7786 | 830* |
| bnb_int8 | bnb LLM.int8() W8A8 | yes | 311 | 0.8092 | +0.0001 | 0.7858 | **+0.0000** | 0.072 | 0.057 | 0.0037 | 0.0050 | 0.009 | 0.99999 | 0.8009 | 0.7783 | 1165 |
| bnb_nf4 | bnb NF4 4-bit wt | yes* | 156 | 0.8088 | −0.0002 | 0.7861 | +0.0003 | 0.411 | 0.600 | 0.0186 | 0.0262 | 0.049 | 0.99980 | 0.8006 | 0.7777 | 840 |
| bnb_fp4 | bnb FP4 4-bit wt | yes* | 156 | 0.8088 | −0.0003 | 0.7852 | −0.0006 | 0.519 | 0.514 | 0.0248 | 0.0340 | 0.065 | 0.99968 | 0.8007 | 0.7780 | 830 |
| fp8_dyn | torchao fp8 W8A8 (bf16) | NO | 311 | 0.8096 | +0.0005 | 0.7852 | −0.0006 | 0.221 | 0.314 | 0.0090 | 0.0122 | 0.024 | 0.99996 | 0.8013 | 0.7777 | 757* |

**✅ Native track COMPLETE (9 rows).**

**TensorRT track** (TRT 10.7, docker 24.12-py3, RTX 4060; drift vs fp16 PyTorch REF):

| method | build s | engine MB | infer s (70k) | rows/s | ΔF1 val | flip% val | logit L1 | verdict |
|---|---|---|---|---|---|---|---|---|
| trt_fp32 | 29 | 1253 | 1096 | 64 | +0.0000 | (≡fp32 sanity) | ~0 | export exact |
| **trt_fp16** | 41 | 628 | **328** | **213** | **−0.0002** | 0.057 | 0.0016 | **1.66× faster than PyTorch fp16 (128 r/s), drift below the bnb_int8 level — cleanest fast path** |
| trt_int8_entropy | 268 | 628 | 329 | 213 | — | — | — | **⚠️ NO-OP**: logits byte-identical to trt_fp16 |
| trt_int8_minmax | 186 | 628 | 330 | 212 | — | — | — | **⚠️ NO-OP**: logits byte-identical to trt_fp16 |

**TRT INT8 finding (implicit):** implicit-quantization INT8 (BuilderFlag.INT8 + IInt8Calibrator) is **deprecated-and-inert in TRT 10.7** — calibration executed (cache written, ~79 KB of scales) but the builder selected zero int8 tactics; engine == fp16 engine, bit-identical outputs.

**TRT INT8 (explicit, modelopt Q/DQ) — real int8, but BROKEN accuracy (2026-07-14):** built a genuine int8 engine via nvidia-modelopt 0.45 entropy PTQ (512 calib rows, CUDA-only calibration EP — the default 'trt' EP OOMs the 8 GB/15 GB box building calibration engines for the 1.25 GB graph). Engine **524 MB < fp16 628 MB** (int8 body confirmed) and **270 rows/s = ~1.27× faster than TRT fp16** (213 r/s) — *acceleration is real*. **BUT accuracy collapsed: F1 val 0.7858 → 0.0283, flip 87%, logit cos 0.148.** Cause = **logit-range collapse**: fp16 logits span ~[−1.3, +4.2] but int8 crush to ~[−1.3, +0.7], predictions pile into 2 classes (85% class-1, 15% class-7; pred-entropy 0.63/3.81 bits). This was a modelopt config defect, diagnosed from its own log: default `op_types_to_quantize` was `['Add','Gemm','Mul','MatMul','LayerNormalization']` + 22 MHA attention matmuls (243 nodes) — quantizing **LayerNorm + residual Adds + attention** collapses transformer range ([ZeroQuant 2206.01861](https://arxiv.org/pdf/2206.01861)); plus a `RuntimeWarning: overflow encountered in cast` from modelopt AutoCast's fp32→fp16 (NVIDIA docs: "AutoCast does not yet support quantized models").

**TRT INT8 (explicit) — FIXED, F1-lossless + accelerated (2026-07-14):** two-param fix — `op_types_to_quantize=["MatMul","Gemm"]` (linear projections only → 132 nodes, no LayerNorm/Add/Mul) + `high_precision_dtype="fp32"` (no autocast → overflow gone). Result: **all 14 classes, ΔF1 val −0.0014 (noise-level), flip 1.11%, logit cos 0.99898, acc val 0.7754 vs 0.7783** — vs the broken run's F1 0.028 / cos 0.148. **Speed 241 rows/s = ~1.13× faster than TRT fp16 (213)**, engine 526 MB. (Below the broken run's 270 r/s because fp32 high-precision fallback on non-quantized layers costs throughput — recoverable with fp16 high-precision + selective node exclusion.) **Verdict: a properly-configured TRT int8 engine DOES accelerate on the T4's int8 tensor cores AND holds F1 — the accelerated-int8 path is viable.** Artifacts: `logits_trt_int8x_entropy.npz`, `trt_int8x_entropy.engine`; reusable image `trt-modelopt:24.12`.

**TRT deployment validation (2026-07-14/15, real T4 via Colab free tier):**
- **TRT fp16 on T4** (engine built on-T4 from the fp32 ONNX): **F1 0.7853, build 84 s, infer 37.2 s/3.5k → projected 30k = 403 s — FITS the 600 s cap.** Pinned `tensorrt==10.7.0` pip-installs on T4 in ~130 s (inside the 10-min install window). Colab gotchas fixed: default pip TRT is bleeding-edge (BuilderFlags removed) → pin 10.7; torch≥2.9 needs `onnxscript` or `dynamo=False` for export.
- **Shippable int8 model** `_work/model_int8_ship.onnx` (**627 MB, fits 1 GB cap**): fp32 ONNX in + `high_precision_dtype='fp16'` out + MatMul/Gemm-only. ⚠️ modelopt quantization needs >12.7 GB RAM (OOM'd Colab) → quantize OFFLINE (done, 19 min local), ship the quantized model, only *build the engine* on target. Note: the fp16→ONNX (onnxconverter) file is NOT modelopt-ingestible (mixed float/float16 `Mul` types error) — feed modelopt the fp32 export.
- **Local F1 gate on the shippable int8** (4060-built engine; accuracy is HW-independent): **F1 val 0.7857 vs ref 0.7858 — lossless**, 14/14 classes, engine 526 MB, build 60 s. Better than the fp32-high-precision variant (0.7844) AND faster.
- **TRT int8 on the REAL T4** (Colab, engine built on-T4): **infer 24.2 s/3.5k = 1.54× faster than T4-fp16** (37.2 s) — **T4 int8 tensor-core acceleration confirmed**. F1 0.7839 (T4 build's tactic choices cost ~0.002 vs the 4060 build's 0.7857). Build 220 s → proj 30k = 427 s (fits cap). Engine saved off Colab → **shipped directly** (same arch + pinned `tensorrt==10.7.0` ⇒ loads on DACON with NO on-box build): **`submit_0715_awp_trt8.zip` (563 MB)** = T4 engine 528 MB + full tokenizer + load-and-run script. Projected DACON ≈ engine load + ~207 s infer ≈ 4–5 min.
- **VOCAB-PRUNED int8 chain (2026-07-15):** E28 remap reused (262,152→54,688 rows; embedding is 64.5% of weights and untouched by quant — pruning is the size lever). Prune parity **exact** (max|Δlogit| 0.000000). Pruned fp32 ONNX 612 MB → modelopt int8 (same recipe) → **`model_int8_pruned.onnx` 308 MB**; local F1 gate **0.7857 — pruning cost ZERO accuracy**, engine 208 MB (was 528). Inference remaps ids (`remap[input_ids]`) after the COMPLETE tokenizer (tokenizer.json is never edited — piece/merge pruning would break tokenization parity). Final package `submit_0715_awp_trt8p.zip` (~245 MB) pending the T4 engine build (Colab).

**Deployment math (DACON):** TRT fp16 does 70k rows in 328 s on the 4060 → 30k test rows ≈ 141 s here; even at T4 ≈ 0.45× 4060 throughput that's ≈ **5:10, comfortably inside the 10-min cap** — TRT fp16 alone already buys the time headroom; INT8 isn't needed for the cap.
`*` torchao latency = reference path, not meaningful; bnb latency = real kernels. bnb `yes*` = runs on T4 but 4-bit is storage-only there (compute dequants to fp16).

---

## T4 inference acceleration — which methods actually speed up (verified 2026-07-14)

**Distinction:** the T4 (Turing SM 7.5) *hardware* HAS fp16 **and** int8 (and int4/int1) tensor cores — but a method only benefits if its **kernel dispatches to them**. Weight-only int8 and all bitsandbytes paths dequantize to fp16 for the matmul, so they never touch the int8 units; they're **memory/size** techniques, not speed ones (confirmed: [LLM.int8() paper App. D.1 — 8-bit is *slower* than fp16 at small dims](https://arxiv.org/pdf/2208.07339), [bnb issue #856](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/856)). Only a real fused int8 GEMM (TensorRT / ONNXRuntime / CUTLASS) uses them.

| method (tried) | T4 int8/fp16 tensor cores used? | speed vs fp16 on T4 | verdict |
|---|---|---|---|
| fp32 | ➖ CUDA cores only | slowest | baseline |
| **bf16** | ⛔ **won't run** | — | Turing has **no bf16** (needs SM ≥8.0); measurement dtype only |
| int8_wo (torchao W8A16) | ❌ no | ≈/slower | weight-only ⇒ fp16 matmul by design; dequant tax |
| int8_dyn (torchao W8A8) | ❌ no (torchao int8 kernels target Ampere+; ref-path here) | slower | *could* accelerate with a proper kernel, not this one |
| bnb_int8 (LLM.int8 W8A8) | ❌ no (dequant + outlier decomposition) | **slower** (measured 2.1×) | **memory only** |
| bnb_nf4 / bnb_fp4 (4-bit) | ❌ no (dequant to fp16) | slower (1.5×) | **storage/size only** |
| **fp8_dyn** (torchao) | ⛔ **won't run** | — | fp8 needs SM ≥8.9 (Ada); T4 has none |
| trt_fp16 | ✅ **yes (fp16 TC)** | **~1.5–2× faster** ✅ | **the measured speed win** (not a quant method) |
| trt_int8 (entropy/minmax) | ✅ *would* (int8 TC) | *would be* ~1.5–2× | ⚠️ **NOT achieved** — TRT 10.7 implicit-quant was a NO-OP (engine == fp16) |

**Bottom line: of the quantization methods we actually tried, NONE deliver T4 acceleration.** The only path that *would* (TensorRT int8) requires **explicit quantization** (nvidia-modelopt Q/DQ, or a TRT 8.6 container) and remains unmeasured → follow-up. The measured speed win (TRT fp16, 1.66× here) is not quantization. **⇒ On this project quantization = a SIZE lever (fit ≤1 GB); the SPEED lever is TRT-fp16 + batch/length-sort.**

### Native-track interim findings (granite)
1. **No PTQ scheme costs granite any F1.** All ΔF1 val within ±0.001 (= slice noise); even 4-bit holds. The AWP champion's loss surface is flat enough that PTQ noise doesn't reach the argmax on ~99.4%+ of rows.
2. **Drift ranking (logit L1): bf16 floor 0.003 < bnb_int8 0.004 < int8_wo 0.007 < fp8 0.009 < nf4 0.019 < int8_dyn 0.023 < fp4 0.025.** Outlier-protected LLM.int8() is the cleanest actual quantization — beats even weight-only RTN. Activation quant (int8_dyn) drifts more than 4-bit weight-only (nf4): on this model **activations are the fragile axis, weights are robust**.
3. **NF4 > FP4** at equal bits (L1 0.019 vs 0.025, cos 0.99980 vs 0.99968) — the distribution-matched grid wins, textbook.
4. **fp8 ≈ int8-quality with no special handling** (L1 0.009) — but Ada-only, not portable to the DACON T4.
5. **Prediction flips concentrate in class 1 (`grep_search`)** in every scheme — quant noise flips the already-ambiguous boundary rows, consistent with the known label-ambiguity ceiling.
6. All drift is label-free-measured; F1 read on train (66.5k in-sample) vs val (3.5k held-out, session-leaky + used for ckpt selection → optimistic; ΔF1 is the trustworthy read).

---

## LEADERBOARD VALIDATION (2026-07-14) — quantization is LB-lossless on the real T4

| submission | LB score | T4 runtime | vs fp16 champion 0.79300 | local ΔF1 val (predicted) |
|---|---|---|---|---|
| `submit_0714_awp_int8.zip` (bnb LLM.int8) | **0.793310** | **5:18** | **+0.00031** | +0.0000 |
| `submit_0714_awp_nf4.zip` (bnb NF4 4-bit) | **0.793072** | **3:33** | +0.00007 | +0.0003 |
| fp16 champion (E38 t031, ref) | 0.79300 | ~5:06 | — | 0 |

**Both quantized zips match/slightly beat the fp16 champion (within ~0.0003 noise) → quantization holds the score on the hidden test set.** The local 70k drift study *correctly predicted* this (both ΔF1 val ≈ 0). Both fit the **10-min cap comfortably**.

⚠️ **Runtime correction:** local 4060 measured bnb int8 at **2.1× slower** than fp16 and nf4 at 1.5× slower — but on the **T4** int8 ran **5:18 (≈ fp16's 5:06, NOT 2×)** and **nf4 ran 3:33 (FASTER than fp16)**. The local slowdown ratios did NOT transfer — T4 memory subsystem + the submission's length-sorted batching change the picture. **Lesson: quant *runtime* is HW-specific; measure on the target (or the LB), don't extrapolate the ratio from a different GPU.** (Accuracy/drift *does* transfer; runtime does not.) The pre-submission timeout worry for int8 was therefore wrong.

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
