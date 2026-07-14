"""E46 — run one PyTorch-native quantization variant, measure drift vs fp16 reference.

Usage: python run_variant.py <method>
methods (T4 column = runs on a T4 SM7.5):
  fp32            fp32 weights, fp32 compute          (T4: yes)  reference-of-reference
  int8_wo         torchao int8 weight-only  W8A16     (T4: yes)
  int8_dyn        torchao int8 dynamic act  W8A8      (T4: yes)
  int4_wo         torchao int4 weight-only  W4A16     (T4: NO, needs SM>=8.0 bf16 tinygemm)
  fp8_dyn         torchao float8 dynamic    W8A8-fp8  (T4: NO, needs SM>=8.9 Ada)
  bnb_int8        bitsandbytes LLM.int8()   W8A8      (T4: yes)
  bnb_nf4         bitsandbytes NF4 4-bit weight       (T4: yes, storage-only)
  bnb_fp4         bitsandbytes FP4 4-bit weight       (T4: yes, storage-only)

Appends a JSON line to _work/results.jsonl and prints a one-line summary.
"""
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

T4_OK = {"fp32": "yes", "int8_wo": "yes", "int8_dyn": "yes", "int4_wo": "NO(SM>=8.0)",
         "fp8_dyn": "NO(SM>=8.9)", "bnb_int8": "yes", "bnb_nf4": "yes-storage", "bnb_fp4": "yes-storage"}


def load_plain(dtype):
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=dtype)
    return m


def apply_torchao(method):
    from torchao.quantization import (quantize_, Int8WeightOnlyConfig,
                                      Int8DynamicActivationInt8WeightConfig,
                                      Int4WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig)
    dtype = torch.bfloat16 if method in ("int4_wo",) else torch.float16
    m = load_plain(dtype).cuda().eval()
    cfg = {
        "int8_wo": Int8WeightOnlyConfig(),
        "int8_dyn": Int8DynamicActivationInt8WeightConfig(),
        "int4_wo": Int4WeightOnlyConfig(group_size=128),
        "fp8_dyn": Float8DynamicActivationFloat8WeightConfig(),
    }[method]
    t0 = time.time()
    quantize_(m, cfg)
    return m, dtype, round(time.time() - t0, 2)


def apply_bnb(method):
    from transformers import BitsAndBytesConfig
    if method == "bnb_int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
    elif method == "bnb_nf4":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
    else:  # bnb_fp4
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4",
                                 bnb_4bit_compute_dtype=torch.float16)
    t0 = time.time()
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, quantization_config=bnb, device_map={"": 0}).eval()
    return m, torch.float16, round(time.time() - t0, 2)


def approx_size_mb(method):
    """Rough on-disk weight footprint of the transformer body under the scheme (622MB fp16 base)."""
    base = 622.0  # fp16 weights of the 311M model
    return {"fp32": base * 2, "int8_wo": base / 2, "int8_dyn": base / 2, "int4_wo": base / 4,
            "fp8_dyn": base / 2, "bnb_int8": base / 2, "bnb_nf4": base / 4, "bnb_fp4": base / 4}.get(method, base)


def main():
    method = sys.argv[1]
    os.chdir(L.REPO)
    ref = np.load(REF, allow_pickle=True)
    ref_logits = ref["logits"].astype(np.float32)
    y_true = ref["y_true"].astype(np.int64)
    val_mask = ref["val_mask"]

    ids, texts, y2, vm2 = L.build_texts()
    assert (y2 == y_true).all() and (vm2 == val_mask).all(), "data/split mismatch vs reference"

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    tok.truncation_side = "right"

    if method == "fp32":
        m = load_plain(torch.float32).cuda().eval()
        ac, qt = torch.float32, 0.0
    elif method.startswith("bnb_"):
        m, ac, qt = apply_bnb(method)
    else:
        m, ac, qt = apply_torchao(method)

    logits, timing = L.run_model(m, tok, texts, autocast_dtype=ac, desc=method)
    met = L.drift_metrics(ref_logits, logits, y_true, val_mask)

    rec = {"method": method, "t4": T4_OK.get(method, "?"), "size_mb": round(approx_size_mb(method), 0),
           "quant_s": qt, **timing, **met}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    np.savez_compressed(os.path.join(L.REPO, f"experiments/compression/quantization/_work/logits_{method}.npz"),
                        logits=logits.astype(np.float16))
    print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
