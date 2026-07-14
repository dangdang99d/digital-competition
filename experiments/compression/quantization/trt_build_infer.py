"""E46 TRT track — build TensorRT engines from model.onnx and run all 70k rows.

Runs INSIDE the nvcr.io/nvidia/tensorrt:24.12-py3 container (TRT 10.x):
  docker run --rm --gpus all -v <repo>:/ws nvcr.io/nvidia/tensorrt:24.12-py3 \
      python /ws/experiments/compression/quantization/trt_build_infer.py <variant>

variants: fp32 | fp16 | int8_entropy | int8_minmax
Outputs per variant (in _work/): trt_<variant>.engine, logits_trt_<variant>.npz,
and a line in trt_results.jsonl (build_s, infer_s, engine_mb).

INT8 uses implicit quantization + calibrator (deprecated in TRT 10 but functional);
calibration set = first 512 rows of tokens_512.npz (batch 64 x 8).
"""
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
BATCH = 64
SEQ = 512
N_CLASSES = 14

logger = trt.Logger(trt.Logger.WARNING)


def _cuda():
    import ctypes
    lib = ctypes.CDLL("libcudart.so")
    lib.cudaMalloc.restype = int
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    return lib


CUDART = _cuda()
import ctypes  # noqa: E402


def cuda_malloc(nbytes):
    p = ctypes.c_void_p()
    assert CUDART.cudaMalloc(ctypes.byref(p), nbytes) == 0
    return p.value


def memcpy_htod(dst, arr):
    assert CUDART.cudaMemcpy(ctypes.c_void_p(dst),
                             arr.ctypes.data_as(ctypes.c_void_p),
                             arr.nbytes, 1) == 0


def memcpy_dtoh(arr, src):
    assert CUDART.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_void_p(src), arr.nbytes, 2) == 0


class Calibrator(trt.IInt8EntropyCalibrator2):
    """Feeds (input_ids, attention_mask) int64 batches for INT8 calibration."""

    def __init__(self, ids, mask, n_batches=8, cache=""):
        super().__init__()
        self.ids, self.mask = ids, mask
        self.n, self.i = n_batches, 0
        self.cache_file = cache
        self.d_ids = cuda_malloc(BATCH * SEQ * 8)
        self.d_mask = cuda_malloc(BATCH * SEQ * 8)

    def get_batch_size(self):
        return BATCH

    def get_batch(self, names):
        if self.i >= self.n:
            return None
        s = self.i * BATCH
        memcpy_htod(self.d_ids, np.ascontiguousarray(self.ids[s:s + BATCH].astype(np.int64)))
        memcpy_htod(self.d_mask, np.ascontiguousarray(self.mask[s:s + BATCH].astype(np.int64)))
        self.i += 1
        return [self.d_ids, self.d_mask]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            return open(self.cache_file, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            open(self.cache_file, "wb").write(cache)


class MinMaxCalibrator(Calibrator, trt.IInt8MinMaxCalibrator):
    def __init__(self, ids, mask, n_batches=8, cache=""):
        trt.IInt8MinMaxCalibrator.__init__(self)
        Calibrator.__init__(self, ids, mask, n_batches, cache)


def build_engine(variant, ids, mask):
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    for name in ("input_ids", "attention_mask"):
        profile.set_shape(name, (1, SEQ), (BATCH, SEQ), (BATCH, SEQ))
    config.add_optimization_profile(profile)

    if variant == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif variant.startswith("int8"):
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # mixed int8+fp16 fallback
        cal_cls = Calibrator if variant == "int8_entropy" else MinMaxCalibrator
        config.int8_calibrator = cal_cls(ids, mask, n_batches=8,
                                         cache=os.path.join(WORK, f"calib_{variant}.cache"))
        # calibration profile required for dynamic shapes
        cprof = builder.create_optimization_profile()
        for name in ("input_ids", "attention_mask"):
            cprof.set_shape(name, (BATCH, SEQ), (BATCH, SEQ), (BATCH, SEQ))
        config.set_calibration_profile(cprof)

    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    build_s = round(time.time() - t0, 1)
    if blob is None:
        raise RuntimeError(f"engine build failed for {variant}")
    path = os.path.join(WORK, f"trt_{variant}.engine")
    with open(path, "wb") as f:
        f.write(blob)
    return path, build_s


def infer_all(engine_path, ids, mask):
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
    ctx = engine.create_execution_context()

    d_ids = cuda_malloc(BATCH * SEQ * 8)
    d_mask = cuda_malloc(BATCH * SEQ * 8)
    d_out = cuda_malloc(BATCH * N_CLASSES * 4)
    out = np.empty((BATCH, N_CLASSES), dtype=np.float32)

    n = len(ids)
    logits = np.empty((n, N_CLASSES), dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        b = e - s
        bi = np.ascontiguousarray(ids[s:e].astype(np.int64))
        bm = np.ascontiguousarray(mask[s:e].astype(np.int64))
        ctx.set_input_shape("input_ids", (b, SEQ))
        ctx.set_input_shape("attention_mask", (b, SEQ))
        memcpy_htod(d_ids, bi)
        memcpy_htod(d_mask, bm)
        ctx.set_tensor_address("input_ids", d_ids)
        ctx.set_tensor_address("attention_mask", d_mask)
        ctx.set_tensor_address("logits", d_out)
        assert ctx.execute_async_v3(0)
        CUDART.cudaDeviceSynchronize()
        memcpy_dtoh(out[:b], d_out)
        logits[s:e] = out[:b]
        if (s // BATCH) % 200 == 0:
            print(f"  {s}/{n}", flush=True)
    infer_s = round(time.time() - t0, 2)
    return logits, infer_s


def main():
    variant = sys.argv[1]
    d = np.load(TOKENS)
    ids, mask = d["input_ids"], d["attention_mask"]
    print(f"tokens: {ids.shape}", flush=True)

    print(f"building {variant} engine...", flush=True)
    path, build_s = build_engine(variant, ids, mask)
    engine_mb = round(os.path.getsize(path) / 1e6, 1)
    print(f"built {path} ({engine_mb} MB) in {build_s}s", flush=True)

    logits, infer_s = infer_all(path, ids, mask)
    np.savez_compressed(os.path.join(WORK, f"logits_trt_{variant}.npz"),
                        logits=logits.astype(np.float16))
    rec = {"method": f"trt_{variant}", "build_s": build_s, "infer_s": infer_s,
           "rows_per_s": round(len(ids) / infer_s, 1), "engine_mb": engine_mb}
    with open(os.path.join(WORK, "trt_results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
