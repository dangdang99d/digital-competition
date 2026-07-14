"""Quick bnb int8 submission build — quantize + save + package. NO parity/smoke gate (check later)."""
import json
import os
import shutil

import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

REPO = "/mnt/nfs/data/research/ocean_backup-dacon"
FP16 = f"{REPO}/experiments/compression/quantization/_work/model/granite-311m-e8a-ls-awp-t031"
V = f"{REPO}/experiments/compression/quantization/_work/build/int8"
M = f"{V}/model/granite-311m-e8a-ls-awp-t031"

shutil.rmtree(V, ignore_errors=True)
os.makedirs(M, exist_ok=True)

print("quantizing load_in_8bit...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    FP16, local_files_only=True, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map={"": 0}).eval()
model.save_pretrained(M)
for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    shutil.copy(f"{FP16}/{fn}", f"{M}/{fn}")
shutil.copy(f"{REPO}/experiments/compression/quantization/pkg_script.py", f"{V}/script.py")
open(f"{V}/requirements.txt", "w").write(
    "torch\ntransformers==4.51.3\nnumpy\nscikit-learn\nbitsandbytes==0.49.2\naccelerate\n")
json.dump({"variant": "richargs"}, open(f"{V}/serialize_variant.json", "w"))
wb = sum(os.path.getsize(f"{M}/{f}") for f in os.listdir(M) if f.endswith(".safetensors"))
print(f"SAVED weights={wb/1e6:.1f}MB dir={V}", flush=True)
