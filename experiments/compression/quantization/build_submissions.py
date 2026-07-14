"""E46 arm A — build bnb-quantized granite submission zips (pre-quantized weights).

For each variant {nf4, int8}:
  1. load fp16 granite → quantize with bnb → save_pretrained (real quantized weights)
  2. copy tokenizer + write script.py / requirements.txt / serialize_variant.json
  3. PARITY GATE: reload the saved quantized model, infer 1024 train rows, compare argmax
     to the measured logits_bnb_<v>.npz — must match (else abort that variant)
  4. zip → submissions/submit_0714_awp_<v>.zip ; report weight + zip sizes
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig)

import lib_infer as L
import pkg_script as PKG

REPO = L.REPO
FP16_DIR = os.path.join(REPO, "experiments/compression/quantization/_work/model/granite-311m-e8a-ls-awp-t031")
WORK = os.path.join(REPO, "experiments/compression/quantization/_work")
BUILD = os.path.join(REPO, "experiments/compression/quantization/_work/build")
MODELNAME = "granite-311m-e8a-ls-awp-t031"

REQS = "torch\ntransformers==4.51.3\nnumpy\nscikit-learn\nbitsandbytes==0.49.2\naccelerate\n"


def bnb_cfg(variant):
    if variant == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16)
    return BitsAndBytesConfig(load_in_8bit=True)


def build(variant):
    print(f"\n===== build {variant} =====", flush=True)
    vdir = os.path.join(BUILD, variant)
    mdir = os.path.join(vdir, "model", MODELNAME)
    if os.path.exists(vdir):
        shutil.rmtree(vdir)
    os.makedirs(mdir, exist_ok=True)

    # 1. quantize + save
    model = AutoModelForSequenceClassification.from_pretrained(
        FP16_DIR, local_files_only=True, quantization_config=bnb_cfg(variant),
        device_map={"": 0}).eval()
    model.save_pretrained(mdir)
    # tokenizer
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(FP16_DIR, fn), os.path.join(mdir, fn))
    del model
    torch.cuda.empty_cache()

    wbytes = sum(os.path.getsize(os.path.join(mdir, f)) for f in os.listdir(mdir)
                 if f.endswith(".safetensors"))
    print(f"  saved quantized weights: {wbytes/1e6:.1f} MB", flush=True)

    # 2. package files
    shutil.copy(os.path.join(os.path.dirname(__file__), "pkg_script.py"),
                os.path.join(vdir, "script.py"))
    open(os.path.join(vdir, "requirements.txt"), "w").write(REQS)
    json.dump({"variant": "richargs"}, open(os.path.join(vdir, "serialize_variant.json"), "w"))

    # 3. PARITY GATE — reload saved quantized model, compare argmax on 1024 train rows
    ids, texts, y, vm = L.build_texts()
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    tok.truncation_side = "right"
    remodel = AutoModelForSequenceClassification.from_pretrained(
        mdir, local_files_only=True, device_map={"": 0}).eval()
    N = 1024
    logits = PKG.predict_logits(texts[:N], remodel, tok, batch_size=128)
    got = logits.argmax(-1)
    ref = np.load(os.path.join(WORK, f"logits_bnb_{variant}.npz"))["logits"][:N].astype(np.float32).argmax(-1)
    match = int((got == ref).sum())
    print(f"  PARITY {variant}: {match}/{N} argmax match vs measured logits_bnb_{variant}", flush=True)
    del remodel
    torch.cuda.empty_cache()
    if match < N * 0.99:
        print(f"  ⚠️ PARITY FAIL {variant} ({match}/{N}) — NOT zipping", flush=True)
        return None

    # 4. zip
    zip_path = os.path.join(REPO, f"submissions/submit_0714_awp_{variant}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    # store mode (-0): quantized weights are high-entropy → deflate is pointless AND
    # pathologically slow over NFS. store keeps zip ~= weights size (well under 1 GB cap).
    subprocess.run(["zip", "-r", "-0", "-q", zip_path, "serialize_variant.json",
                    "requirements.txt", "script.py", "model"], cwd=vdir, check=True)
    zbytes = os.path.getsize(zip_path)
    zname = os.path.basename(zip_path)
    print(f"  ZIP {zname} ({len(zname)} chars): {zbytes/1e6:.1f} MB", flush=True)
    return {"variant": variant, "weights_mb": round(wbytes/1e6, 1),
            "zip_mb": round(zbytes/1e6, 1), "zip": zname, "namelen": len(zname), "parity": match}


def main():
    os.chdir(REPO)
    os.makedirs(BUILD, exist_ok=True)
    out = []
    for v in (sys.argv[1:] or ["nf4", "int8"]):
        r = build(v)
        if r:
            out.append(r)
    print("\n=== SUMMARY ===")
    for r in out:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
