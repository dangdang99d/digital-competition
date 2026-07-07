"""E16 — ShortGPT Block-Influence layer selection for qwen3-0.6B (28 layers).

Faithful to icip-cas/ShortGPT `shortgpt/metrics.py` block_influence():
    BI_i = 1 - cos(h_in_i, h_out_i)   averaged over calibration tokens
where h_in_i = hidden_states[i] (input to layer i), h_out_i = hidden_states[i+1]
(output of layer i). HF returns hidden_states of length num_layers+1 with [0] = embeddings.
Drop the 14 LOWEST-BI layers (highest input->output cosine = most redundant / identity-like);
keep the other 14.

Minor faithful divergence vs official code: official reshapes ALL positions (incl. pad).
We mask to non-pad positions via attention_mask (official ShortGPT runs unpadded stride
windows in causal LM; padding would pollute cosine here). Ranking is sum vs mean invariant.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.insert(0, "/home/ocean/dacon")
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from src.data import load_samples, build_texts, split_indices, CLASS_TO_ID

CKPT = "/home/ocean/dacon/output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e8b_qwen3_richargs_full/checkpoint-8314"
N_CALIB = 300
MAX_LEN = 512
BS = 8
N_KEEP = 14
device = "cuda"

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
tok.truncation_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    CKPT, torch_dtype=torch.float32, trust_remote_code=True)
model.config.pad_token_id = tok.pad_token_id
model.to(device).eval()
L = model.config.num_hidden_layers
print(f"model depth L={L}  hidden={model.config.hidden_size}")

# same split as E8b/E4 (seed 42 + full_data 25% eval carve-out)
samples, y = load_samples("/home/ocean/dacon/data")
texts = build_texts(samples, input_mode="context", max_hist=None, variant="richargs")
y_ids = np.array([CLASS_TO_ID[a] for a in y])
tr, va = split_indices(y, seed=42)
_, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
calib_idx = va_eval[:N_CALIB]
calib_texts = [texts[i] for i in calib_idx]
print(f"calibration: {len(calib_texts)} richargs-serialized val-eval samples")

bi_sum = np.zeros(L, dtype=np.float64)
tok_count = 0
with torch.no_grad():
    for start in range(0, len(calib_texts), BS):
        batch = calib_texts[start:start + BS]
        enc = tok(batch, truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple, len L+1
        mask = enc["attention_mask"].bool().reshape(-1)
        for i in range(L):
            d = hs[i].shape[-1]
            inp = hs[i].reshape(-1, d)[mask]
            outp = hs[i + 1].reshape(-1, d)[mask]
            cos = torch.nn.functional.cosine_similarity(inp, outp, dim=-1)
            bi_sum[i] += (1.0 - cos).sum().item()
        tok_count += int(mask.sum().item())

bi = bi_sum / tok_count
order_low = np.argsort(bi)          # ascending BI = most redundant first
drop = sorted(order_low[:L - N_KEEP].tolist())
keep = sorted(set(range(L)) - set(drop))

print("\n=== Block-Influence per layer (mean 1-cos over calib tokens) ===")
for i in range(L):
    tag = "DROP" if i in drop else "keep"
    print(f"  layer {i:2d}: BI={bi[i]:.5f}  [{tag}]")
print(f"\ntokens={tok_count}")
print(f"DROP (14 lowest-BI): {drop}")
print(f"KEEP (14 highest-BI): {keep}")
print(f"KEEP_CSV={','.join(map(str, keep))}")

os.makedirs("/home/ocean/dacon/experiments/pruning", exist_ok=True)
np.save("/home/ocean/dacon/experiments/pruning/e16_bi.npy", bi)
with open("/home/ocean/dacon/experiments/pruning/e16_keep.txt", "w") as f:
    f.write(",".join(map(str, keep)) + "\n")
