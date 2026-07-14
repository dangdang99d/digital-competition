"""DACON inference — TensorRT fp16 engine, built ON THE TARGET GPU at runtime.

Ships an fp16 ONNX (model/model_fp16.onnx); this script builds the TRT engine for
whatever GPU it runs on (T4 on DACON), then runs length-sorted inference. TRT engines
are arch-specific, so the engine MUST be built here, not shipped. No torch needed.
richargs serialization; raw argmax over 14 classes; no logit calibration.
"""
import csv
import ctypes
import json
import os
import time
from pathlib import Path

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
SEQ, NCLS = 512, 14


# ---------- richargs serialization (byte-identical to training) ----------
def _budget_bucket(t):
    return "very_low" if t < 2000 else "low" if t < 10000 else "medium" if t < 50000 else "high"


def _elapsed_bucket(s):
    return "early" if s < 120 else "mid" if s < 900 else "late"


def _strip_dirs(v):
    if "/" not in v:
        return v
    tail = "/" if v.endswith("/") else ""
    return v.rstrip("/").rsplit("/", 1)[-1] + tail


def serialize_richargs(r):
    sm = r["session_meta"]; ws = sm["workspace"]; parts = []
    mix = ws.get("language_mix") or {}
    codelang = max(mix.items(), key=lambda kv: kv[1])[0] if mix else "-"
    names = [f.rsplit("/", 1)[-1] for f in ws["open_files"][:6]]
    parts.append(
        f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
        f"budget={_budget_bucket(sm['budget_tokens_remaining'])} "
        f"elapsed={_elapsed_bucket(sm['elapsed_session_sec'])} "
        f"codelang={codelang} loc={ws.get('loc', '-')} ci={ws['last_ci_status']} "
        f"dirty={ws['git_dirty']} open={len(ws['open_files'])} files={','.join(names) or '-'}]")
    for t in r["history"]:
        if t.get("role") == "user":
            parts.append(f"USER: {t['content']}")
        else:
            args = {k: _strip_dirs(v) if isinstance(v, str) else v
                    for k, v in (t.get("args", {}) or {}).items()}
            parts.append(f"ACTION {t['name']}({args}) -> {t.get('result_summary', '')}")
    parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


# ---------- cuda memcpy via ctypes (no pycuda dependency) ----------
CUDART = ctypes.CDLL("libcudart.so")
CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]


def cuda_malloc(n):
    p = ctypes.c_void_p(); assert CUDART.cudaMalloc(ctypes.byref(p), n) == 0; return p.value


def h2d(dst, arr):
    assert CUDART.cudaMemcpy(ctypes.c_void_p(dst), arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, 1) == 0


def d2h(arr, src):
    assert CUDART.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src), arr.nbytes, 2) == 0


def build_engine(onnx_path, batch, logger):
    import tensorrt as trt
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), "ONNX parse failed"
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for name in ("input_ids", "attention_mask"):
        profile.set_shape(name, (1, SEQ), (batch, SEQ), (batch, SEQ))
    config.add_optimization_profile(profile)
    return builder.build_serialized_network(network, config)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np
    import tensorrt as trt
    from transformers import AutoTokenizer

    data_dir = Path("./data"); model_root = Path("./model")
    out_path = Path("./output/submission.csv"); batch = 64
    logger = trt.Logger(trt.Logger.WARNING)

    model_dir = sorted(model_root.glob("granite-311m-e8a-ls*"))[0]
    onnx_path = str(model_root / "model_fp16.onnx")
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tok.truncation_side = "right"

    samples = [json.loads(l) for l in open(data_dir / "test.jsonl", encoding="utf-8") if l.strip()]
    ids = [s["id"] for s in samples]
    texts = [serialize_richargs(s) for s in samples]

    print("building TRT fp16 engine on this GPU...", flush=True)
    t0 = time.time()
    blob = build_engine(onnx_path, batch, logger)
    assert blob is not None, "engine build failed"
    engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
    ctx = engine.create_execution_context()
    print(f"engine built in {time.time()-t0:.1f}s", flush=True)

    d_ids, d_mask, d_out = cuda_malloc(batch*SEQ*8), cuda_malloc(batch*SEQ*8), cuda_malloc(batch*NCLS*4)
    out = np.empty((batch, NCLS), np.float32)
    logits = np.empty((len(texts), NCLS), np.float32)

    # length-sorted batching
    tok_lens = [len(tok.encode(t, truncation=True, max_length=SEQ)) for t in texts]
    order = np.argsort(tok_lens, kind="stable")
    t0 = time.time()
    for i in range(0, len(order), batch):
        idx = order[i:i+batch]; b = len(idx)
        enc = tok([texts[j] for j in idx], truncation=True, max_length=SEQ,
                  padding="max_length", return_tensors="np")
        h2d(d_ids, np.ascontiguousarray(enc["input_ids"].astype(np.int64)))
        h2d(d_mask, np.ascontiguousarray(enc["attention_mask"].astype(np.int64)))
        ctx.set_input_shape("input_ids", (b, SEQ)); ctx.set_input_shape("attention_mask", (b, SEQ))
        ctx.set_tensor_address("input_ids", d_ids); ctx.set_tensor_address("attention_mask", d_mask)
        ctx.set_tensor_address("logits", d_out)
        assert ctx.execute_async_v3(0); CUDART.cudaDeviceSynchronize()
        d2h(out[:b], d_out)
        logits[idx] = out[:b]
    print(f"inference {time.time()-t0:.1f}s for {len(texts)} rows", flush=True)

    id2label = {i: c for i, c in enumerate(ALL_CLASSES)}
    preds = {ids[i]: id2label[int(logits[i].argmax())] for i in range(len(ids))}
    with open(data_dir / "sample_submission.csv", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f); fieldnames = rd.fieldnames; rows = list(rd)
    for row in rows:
        row["action"] = preds[row["id"]]
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    print(f"Saved {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
