"""E46 — run ALL PyTorch-native quant variants back-to-back over 70k rows.

Serializes once, loads reference once, then for each method: (re)load+quantize model,
run inference, measure pred/logit drift vs fp16 reference, append to _work/results.jsonl.
Memory freed between methods. Batch bumped (VRAM headroom) for speed; padding is masked
so per-row logits are batch-invariant.

Order runs T4-valid methods first so the deployment-relevant numbers land early.
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import lib_infer as L

MODEL_DIR = os.path.join(L.REPO, "experiments/compression/quantization/_work/model/granite-311m-e8a-ls-awp-t031")
REF = os.path.join(L.REPO, "experiments/compression/quantization/_work/ref_fp16.npz")
RESULTS = os.path.join(L.REPO, "experiments/compression/quantization/_work/results.jsonl")
BATCH = 128

# (method, t4_portability). Ordered: T4-valid first, Ada-only last.
# ⚠️ torchao methods run in bf16: fp16 compute overflows the reference dequant path
#    (no compiled kernels on torch 2.7) → NaN logits. bf16 verified clean on probe.
#    "bf16" plain baseline isolates the dtype-delta from the quant-delta.
METHODS = [
    ("fp32", "yes"),
    ("bf16", "yes(no-TC-accel)"),
    ("int8_wo", "yes"),
    ("int8_dyn", "yes"),
    ("bnb_int8", "yes"),
    ("bnb_nf4", "yes-storage"),
    ("bnb_fp4", "yes-storage"),
    # int4_wo (torchao) dropped: needs mslk kernel not installed, and non-portable to T4
    ("fp8_dyn", "NO(SM>=8.9)"),
]

SIZE_MB = {"fp32": 1244, "bf16": 622, "int8_wo": 311, "int8_dyn": 311, "bnb_int8": 311,
           "bnb_nf4": 156, "bnb_fp4": 156, "int4_wo": 156, "fp8_dyn": 311}


def build_model(method):
    """Return (model, autocast_dtype, quant_seconds)."""
    if method == "fp32":
        m = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True, torch_dtype=torch.float32).cuda().eval()
        return m, torch.float32, 0.0

    if method == "bf16":
        m = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
        return m, torch.bfloat16, 0.0

    if method.startswith("bnb_"):
        from transformers import BitsAndBytesConfig
        if method == "bnb_int8":
            cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            qt = "nf4" if method == "bnb_nf4" else "fp4"
            cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=qt,
                                     bnb_4bit_compute_dtype=torch.float16)
        t0 = time.time()
        m = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True, quantization_config=cfg,
            device_map={"": 0}).eval()
        return m, torch.float16, round(time.time() - t0, 2)

    # torchao — ALWAYS bf16: fp16 overflows the uncompiled reference dequant path (NaN)
    from torchao.quantization import (quantize_, Int8WeightOnlyConfig,
                                      Int8DynamicActivationInt8WeightConfig,
                                      Int4WeightOnlyConfig,
                                      Float8DynamicActivationFloat8WeightConfig)
    dtype = torch.bfloat16
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=dtype).cuda().eval()
    cfg = {"int8_wo": Int8WeightOnlyConfig(),
           "int8_dyn": Int8DynamicActivationInt8WeightConfig(),
           "int4_wo": Int4WeightOnlyConfig(group_size=128),
           "fp8_dyn": Float8DynamicActivationFloat8WeightConfig()}[method]
    t0 = time.time()
    quantize_(m, cfg)
    return m, dtype, round(time.time() - t0, 2)


def main():
    os.chdir(L.REPO)
    only = sys.argv[1:] or None  # optional subset of method names

    ref = np.load(REF, allow_pickle=True)
    ref_logits = ref["logits"].astype(np.float32)
    y_true = ref["y_true"].astype(np.int64)
    val_mask = ref["val_mask"]
    print(f"ref loaded | acc_all={float(ref['acc_all']):.5f} acc_val={float(ref['acc_val']):.5f}",
          flush=True)

    ids, texts, y2, vm2 = L.build_texts()
    assert (y2 == y_true).all() and (vm2 == val_mask).all()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    tok.truncation_side = "right"

    for method, t4 in METHODS:
        if only and method not in only:
            continue
        print(f"\n===== {method} (t4={t4}) =====", flush=True)
        try:
            model, ac, qt = build_model(method)
            logits, timing = L.run_model(model, tok, texts, batch=BATCH, autocast_dtype=ac, desc=method)
            met = L.drift_metrics(ref_logits, logits, y_true, val_mask)
            rec = {"method": method, "t4": t4, "size_mb": SIZE_MB.get(method), "quant_s": qt,
                   **timing, **met}
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            np.savez_compressed(
                os.path.join(L.REPO, f"experiments/compression/quantization/_work/logits_{method}.npz"),
                logits=logits.astype(np.float16))
            print("RESULT " + json.dumps(rec), flush=True)
            del model
        except Exception as e:
            import traceback
            print(f"FAILED {method}: {e}", flush=True)
            traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()

    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
