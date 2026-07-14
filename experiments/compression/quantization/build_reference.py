"""E46 — build the fp16 reference (shipped pipeline) over all 70k rows.

Saves _work/ref_fp16.npz with: logits, pred, y_true, val_mask, ids.
Also prints reference accuracy (all + 3.5k val) as the anchor everything is measured against.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import lib_infer as L

MODEL_DIR = os.path.join(L.REPO, "experiments/compression/quantization/_work/model/granite-311m-e8a-ls-awp-t031")
OUT = os.path.join(L.REPO, "experiments/compression/quantization/_work/ref_fp16.npz")


def main():
    os.chdir(L.REPO)
    print("loading data + serializing (richargs)...", flush=True)
    ids, texts, y_true, val_mask = L.build_texts()
    print(f"n={len(texts)} | val={int(val_mask.sum())} | maxlen chars={max(len(t) for t in texts)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    tok.truncation_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float16).cuda().eval()

    logits, timing = L.run_model(model, tok, texts, autocast_dtype=torch.float16, desc="fp16-ref")
    pred = logits.argmax(-1)
    acc_all = float((pred == y_true).mean())
    acc_val = float((pred[val_mask] == y_true[val_mask]).mean())
    print(f"REF fp16 | acc_all={acc_all:.5f} acc_val={acc_val:.5f} | {timing}", flush=True)

    np.savez_compressed(OUT, logits=logits.astype(np.float16), pred=pred.astype(np.int16),
                        y_true=y_true.astype(np.int16), val_mask=val_mask,
                        ids=np.array(ids), acc_all=acc_all, acc_val=acc_val,
                        infer_s=timing["infer_s"], rows_per_s=timing["rows_per_s"],
                        peak_vram_mb=timing["peak_vram_mb"])
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
