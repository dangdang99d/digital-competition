"""E46 arm A — REAL TensorRT INT8 via EXPLICIT quantization (modelopt Q/DQ).

The implicit-calibrator path was inert in TRT 10.7. This inserts Q/DQ nodes into the
ONNX with modelopt PTQ calibration, then TRT builds a genuine int8 tensor-core engine.

Runs INSIDE nvcr.io/nvidia/tensorrt:24.12-py3 (modelopt pip-installed by the runner):
  python trt_int8_explicit.py <entropy|minmax|max>

Steps (each prints a STEP marker): quantize ONNX -> build engine -> infer 70k -> save logits.
"""
import ctypes
import json
import os
import sys
import time

import numpy as np
import tensorrt as trt

WS = "/ws"
WORK = os.path.join(WS, "experiments/compression/quantization/_work")
ONNX = os.path.join(WORK, "model.onnx")
TOKENS = os.path.join(WORK, "tokens_512.npz")
BATCH, SEQ, NCLS = 64, 512, 14
logger = trt.Logger(trt.Logger.WARNING)

CUDART = ctypes.CDLL("libcudart.so")
CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]


def cuda_malloc(n):
    p = ctypes.c_void_p()
    assert CUDART.cudaMalloc(ctypes.byref(p), n) == 0
    return p.value


def h2d(dst, arr):
    assert CUDART.cudaMemcpy(ctypes.c_void_p(dst), arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, 1) == 0


def d2h(arr, src):
    assert CUDART.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src), arr.nbytes, 2) == 0


def quantize_onnx(method, ids, mask):
    print(f"STEP quantize: modelopt PTQ ({method}) on 512 calib rows", flush=True)
    from modelopt.onnx.quantization import quantize
    out = os.path.join(WORK, f"model_int8_{method}.onnx")
    calib = {"input_ids": ids[:512].astype(np.int64),
             "attention_mask": mask[:512].astype(np.int64)}
    quantize(
        onnx_path=ONNX,
        calibration_data=calib,
        output_path=out,
        quantize_mode="int8",
        calibration_method=method,
        calibration_eps=["cuda:0"],   # CUDA EP only — default 'trt' EP OOMs the 8GB/15GB box
        op_types_to_quantize=["MatMul", "Gemm"],  # linear projections ONLY; exclude
                                       # LayerNorm/Add/Mul (default quantized them → logit-range collapse)
        high_precision_dtype="fp32",  # no fp16 autocast → kills the 'overflow encountered in cast'
    )
    print(f"STEP quantize DONE -> {out} ({os.path.getsize(out)/1e6:.0f} MB)", flush=True)
    return out


def build_engine(qdq_onnx, method):
    print("STEP build: TRT engine from Q/DQ ONNX", flush=True)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(qdq_onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.INT8)   # explicit: Q/DQ drive int8, no calibrator
    config.set_flag(trt.BuilderFlag.FP16)   # fp16 for non-quantized layers
    profile = builder.create_optimization_profile()
    for name in ("input_ids", "attention_mask"):
        profile.set_shape(name, (1, SEQ), (BATCH, SEQ), (BATCH, SEQ))
    config.add_optimization_profile(profile)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("engine build failed")
    bs = round(time.time() - t0, 1)
    path = os.path.join(WORK, f"trt_int8x_{method}.engine")
    open(path, "wb").write(blob)
    print(f"STEP build DONE -> {path} ({os.path.getsize(path)/1e6:.1f} MB, {bs}s)", flush=True)
    return path, bs


def infer_all(engine_path, ids, mask):
    print("STEP infer: 70k rows", flush=True)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
    ctx = engine.create_execution_context()
    d_ids, d_mask, d_out = cuda_malloc(BATCH*SEQ*8), cuda_malloc(BATCH*SEQ*8), cuda_malloc(BATCH*NCLS*4)
    out = np.empty((BATCH, NCLS), np.float32)
    n = len(ids)
    logits = np.empty((n, NCLS), np.float32)
    t0 = time.time()
    for s in range(0, n, BATCH):
        e = min(s+BATCH, n); b = e-s
        h2d(d_ids, np.ascontiguousarray(ids[s:e].astype(np.int64)))
        h2d(d_mask, np.ascontiguousarray(mask[s:e].astype(np.int64)))
        ctx.set_input_shape("input_ids", (b, SEQ))
        ctx.set_input_shape("attention_mask", (b, SEQ))
        ctx.set_tensor_address("input_ids", d_ids)
        ctx.set_tensor_address("attention_mask", d_mask)
        ctx.set_tensor_address("logits", d_out)
        assert ctx.execute_async_v3(0)
        CUDART.cudaDeviceSynchronize()
        d2h(out[:b], d_out)
        logits[s:e] = out[:b]
    infer_s = round(time.time()-t0, 2)
    print(f"STEP infer DONE {infer_s}s ({n/infer_s:.1f} rows/s)", flush=True)
    return logits, infer_s


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "entropy"
    d = np.load(TOKENS)
    ids, mask = d["input_ids"], d["attention_mask"]
    qdq = quantize_onnx(method, ids, mask)
    engine, bs = build_engine(qdq, method)
    logits, infer_s = infer_all(engine, ids, mask)
    np.savez_compressed(os.path.join(WORK, f"logits_trt_int8x_{method}.npz"),
                        logits=logits.astype(np.float16))
    rec = {"method": f"trt_int8x_{method}", "build_s": bs, "infer_s": infer_s,
           "rows_per_s": round(len(ids)/infer_s, 1), "engine_mb": round(os.path.getsize(engine)/1e6, 1)}
    open(os.path.join(WORK, "trt_results.jsonl"), "a").write(json.dumps(rec) + "\n")
    print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
