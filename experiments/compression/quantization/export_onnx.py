"""E46 TRT track — export the granite/ModernBERT classifier to ONNX (CPU) and cache
tokenized inputs for engine inference + INT8 calibration.

Outputs (in _work/):
  model.onnx           fp32 ONNX graph, dynamic batch axis, fixed seq len via inputs
  tokens_512.npz       input_ids int32 [70000, 512], attention_mask uint8, lengths
ModernBERT export notes: attn_implementation="sdpa" (no flash-attn unpadding path),
reference_compile=False. Padded positions are masked, so fixed-512 padding is
numerically identical to the shipped per-batch dynamic padding.
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only — GPU busy with native sweep

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import lib_infer as L

MODEL_DIR = os.path.join(L.REPO, "experiments/compression/quantization/_work/model/granite-311m-e8a-ls-awp-t031")
WORK = os.path.join(L.REPO, "experiments/compression/quantization/_work")


def main():
    os.chdir(L.REPO)

    # ---- 1. tokenize all 70k once (fixed 512, right-truncation as shipped) ----
    tok_npz = os.path.join(WORK, "tokens_512.npz")
    if not os.path.exists(tok_npz):
        print("serializing + tokenizing 70k rows...", flush=True)
        ids, texts, y, vm = L.build_texts()
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        tok.truncation_side = "right"
        enc = tok(texts, truncation=True, max_length=L.MAX_LEN,
                  padding="max_length", return_tensors="np")
        np.savez_compressed(tok_npz,
                            input_ids=enc["input_ids"].astype(np.int32),
                            attention_mask=enc["attention_mask"].astype(np.uint8))
        print(f"saved {tok_npz} shape={enc['input_ids'].shape}", flush=True)
    else:
        print("tokens_512.npz exists, skipping", flush=True)

    # ---- 2. ONNX export (fp32, CPU, sdpa attention, no compile) ----
    onnx_path = os.path.join(WORK, "model.onnx")
    print("loading model fp32/CPU for export...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa", reference_compile=False).eval()

    d = np.load(tok_npz)
    ii = torch.tensor(d["input_ids"][:2].astype(np.int64))
    am = torch.tensor(d["attention_mask"][:2].astype(np.int64))

    with torch.no_grad():
        ref_out = model(input_ids=ii, attention_mask=am).logits
    print("eager sanity logits[0,:4] =", ref_out[0, :4].numpy(), flush=True)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    print("exporting ONNX (opset 17)...", flush=True)
    torch.onnx.export(
        Wrapper(model), (ii, am), onnx_path,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True)
    sz = os.path.getsize(onnx_path) / 1e6
    print(f"saved {onnx_path} ({sz:.0f} MB)", flush=True)

    # ---- 3. parity check with onnxruntime if available (optional) ----
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        out = sess.run(["logits"], {"input_ids": ii.numpy().astype(np.int64),
                                    "attention_mask": am.numpy().astype(np.int64)})[0]
        diff = np.abs(out - ref_out.numpy()).max()
        print(f"onnxruntime parity max|Δ| = {diff:.6f}", flush=True)
    except ImportError:
        print("onnxruntime not installed — parity check deferred to TRT container", flush=True)

    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
