"""Prune AWP-t031's embedding with the E28 remap (54,688 rows) -> export pruned fp32 ONNX.
Also writes remapped token bundles (calibration + val) for quantize/gate/Colab.
CPU-only; GPU not needed. Reuses the E28 remap per [[reuse-pruned-tokenizer]].
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

REPO = "/mnt/nfs/data/research/ocean_backup-dacon"
W = f"{REPO}/experiments/compression/quantization/_work"
SRC = f"{W}/model/granite-311m-e8a-ls-awp-t031"
REMAP = f"{REPO}/output/optuna/build_e28_trio/model/remap.npy"

remap = np.load(REMAP).astype(np.int64)          # old_id -> new_id (dropped ids -> unk row)
V_new = int(remap.max()) + 1
print(f"remap loaded: {len(remap)} -> {V_new} rows", flush=True)

# inverse map: for each new row, one old id that feeds it (first occurrence)
inv = np.full(V_new, -1, dtype=np.int64)
for old, new in enumerate(remap):               # first-occurrence wins
    if inv[new] == -1:
        inv[new] = old
assert (inv >= 0).all()

print("loading model fp32...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    SRC, local_files_only=True, torch_dtype=torch.float32,
    attn_implementation="sdpa", reference_compile=False).eval()

emb = model.model.embeddings.tok_embeddings
new_emb = torch.nn.Embedding(V_new, emb.embedding_dim, padding_idx=None)
with torch.no_grad():
    new_emb.weight.copy_(emb.weight[torch.from_numpy(inv)])
model.model.embeddings.tok_embeddings = new_emb
model.config.vocab_size = V_new
print(f"embedding pruned: {emb.num_embeddings} -> {V_new}", flush=True)

# sanity: pruned model + remapped ids == original logits (2 rows)
d = np.load(f"{W}/tokens_512.npz")
ii = d["input_ids"][:2].astype(np.int64); am = d["attention_mask"][:2].astype(np.int64)
orig = AutoModelForSequenceClassification.from_pretrained(
    SRC, local_files_only=True, torch_dtype=torch.float32,
    attn_implementation="sdpa", reference_compile=False).eval()
with torch.no_grad():
    lo_o = orig(input_ids=torch.tensor(ii), attention_mask=torch.tensor(am)).logits
    lo_p = model(input_ids=torch.tensor(remap[ii]), attention_mask=torch.tensor(am)).logits
diff = (lo_o - lo_p).abs().max().item()
print(f"PRUNE_PARITY max|dlogit| = {diff:.6f} (want ~0)", flush=True)
assert diff < 1e-3, "prune parity failed"
del orig

# export ONNX
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, input_ids, attention_mask):
        return s.m(input_ids=input_ids, attention_mask=attention_mask).logits

print("exporting pruned ONNX...", flush=True)
torch.onnx.export(Wrap(model), (torch.tensor(remap[ii]), torch.tensor(am)),
    f"{W}/model_pruned.onnx",
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "b"}, "attention_mask": {0: "b"}, "logits": {0: "b"}},
    opset_version=17, do_constant_folding=True)
print(f"ONNX {os.path.getsize(f'{W}/model_pruned.onnx')/1e6:.0f}MB", flush=True)

# remapped token bundles: calibration (512) + full 70k (for gate) + val bundle (for Colab)
full_ids = remap[d["input_ids"].astype(np.int64)].astype(np.int32)
np.savez_compressed(f"{W}/tokens_512_pruned.npz",
                    input_ids=full_ids, attention_mask=d["attention_mask"])
v = np.load(f"{W}/val_tokens.npz")
np.savez_compressed(f"{W}/val_tokens_pruned.npz",
                    val_ids=remap[v["val_ids"].astype(np.int64)].astype(np.int32),
                    val_mask=v["val_mask"], val_y=v["val_y"],
                    calib_ids=remap[v["calib_ids"].astype(np.int64)].astype(np.int32),
                    calib_mask=v["calib_mask"])
print("EXPORT_DONE", flush=True)
