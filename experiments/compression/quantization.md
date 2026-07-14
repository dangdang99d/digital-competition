# Quantization — lower numerical precision (fp16 / int8)

Index: [results.md](results.md).

| Precision | Status | Size | Speed |
|---|---|---|---|
| **fp16** | ✅ **IN USE** — all E26 members packaged fp16, bs256, LB-verified (trio 0.78719) | ~2× | ✅ |
| **int8 (TensorRT)** | 🏃 being explored in a separate session (user 2026-07-14) | ~4× weights | ~1.5–2× if T4 tensor cores hit |
| int4 / lower | not planned | — | — |

## int8 via TensorRT — deployment facts (established 2026-07-14)

- **T4 HAS int8 tensor cores** (Turing SM 7.5, ~130 TOPS int8 vs ~65 TFLOPS fp16). PyTorch's
  native quantization is a **CPU backend** and does NOT use them — you need a compiler
  (TensorRT / ONNX-Runtime) to emit int8 kernels that hit the tensor cores.
- **DACON runtime:** T4 (16GB), 3 vCPU, 12GB RAM; ≤10 min inference; ≤10 min package install
  **with internet** (so we can `pip install tensorrt` ourselves — it isn't pre-installed);
  offline at run time; zip ≤ 1GB.
- **Hard mechanic:** a TensorRT engine is compiled for one GPU arch + TRT version. A **T4
  (SM 7.5) engine must be built on a T4** — a 4060 (SM 8.9) / 3090 (SM 8.6) cannot produce a
  runnable T4 engine. So either build at DACON runtime (ship ONNX + calib cache) or prebuild
  on a rented T4.
- **This task is compute-bound** (short-seq encoder), so int8 is a real *speed* lever (unlike
  weight-only int for decode). But granite fp16 already runs ~5:06 < the 10-min budget, so
  int8's value is the 본선 speed score + ensemble headroom, and it competes with structured
  pruning (which needs no TRT/engine/T4-build).
- **Local-box caveat (ArchServer 4060):** validates int8 *accuracy* (hardware-independent
  delta) + speed ratio only; cannot build the shippable T4 engine.

## Status
fp16 is the deployed base. int8 = separate session. This file tracks the deployment facts;
the accuracy/speed numbers land wherever that session records them.
