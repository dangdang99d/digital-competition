"""E46 quantization — shared inference + drift-metric core.

Loads all 70k labeled rows (data/train.jsonl), serializes them with the SHIPPED
richargs variant (byte-identical to t031 training), reproduces the exact 3.5k
val split (seed=42, two-stage stratified), and provides:
  - build_texts()            -> ids, texts, y_true_ids, val_mask
  - run_model(model, tok)    -> logits [N,14] fp32  (shipped fp16+autocast pipeline)
  - drift_metrics(ref, q)    -> dict of prediction + logit + prob drift vs reference
"""
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

REPO = "/mnt/nfs/data/research/ocean_backup-dacon"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.data import load_samples, split_indices, serialize, CLASS_TO_ID  # noqa: E402

MAX_LEN = 512
BATCH = 64
N_CLASSES = 14


def build_texts():
    """Return (ids, texts, y_true_ids [N], val_mask [N] bool). Order == train.jsonl."""
    samples, y = load_samples("data")
    ids = [s["id"] for s in samples]
    y_ids = np.array([CLASS_TO_ID[a] for a in y], dtype=np.int64)

    # richargs == serialize(rich_meta=True, arg_basenames=True); matches training/script.py
    texts = [serialize(r, rich_meta=True, arg_basenames=True) for r in samples]

    # exact t031 full_data 3.5k holdout: stage1 80/20 stratified, stage2 75/25 of the 20%
    from sklearn.model_selection import train_test_split
    tr, va = split_indices(y, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    val_mask = np.zeros(len(samples), dtype=bool)
    val_mask[va_eval] = True
    return ids, texts, y_ids, val_mask


def run_model(model, tokenizer, texts, batch=BATCH, autocast_dtype=None, desc=""):
    """Run the shipped inference pipeline; return logits [N,14] fp32 and timing/vram.

    autocast_dtype: torch dtype for cuda autocast (default float16, matching script.py).
                    Pass torch.float32-equivalent by setting to None AND model in fp32.
    """
    import torch
    model.eval()
    device = next(model.parameters()).device
    if autocast_dtype is None:
        autocast_dtype = torch.float16
    logits_all = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(texts[i:i + batch], truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    out = model(**enc).logits
            else:
                out = model(**enc).logits
            logits_all.append(out.float().cpu().numpy())
            if desc and (i // batch) % 200 == 0:
                print(f"  [{desc}] {i}/{len(texts)}", flush=True)
    dt = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    logits = np.concatenate(logits_all, 0).astype(np.float32)
    return logits, {"infer_s": round(dt, 2),
                    "rows_per_s": round(len(texts) / dt, 1),
                    "peak_vram_mb": round(peak_vram, 1)}


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def drift_metrics(ref_logits, q_logits, y_true, val_mask):
    """Prediction drift + logit drift + prob drift of q vs ref. All raw (no calibration)."""
    ref_pred = ref_logits.argmax(-1)
    q_pred = q_logits.argmax(-1)
    flips = ref_pred != q_pred

    d = q_logits - ref_logits                       # raw logit deltas
    logit_l1 = float(np.abs(d).mean())              # mean |Δ| per element
    logit_rmse = float(np.sqrt((d ** 2).mean()))
    logit_max = float(np.abs(d).max(-1).mean())     # mean over rows of per-row max |Δ|
    # cosine similarity between logit vectors, per row, averaged
    num = (ref_logits * q_logits).sum(-1)
    den = np.linalg.norm(ref_logits, axis=-1) * np.linalg.norm(q_logits, axis=-1) + 1e-9
    logit_cos = float((num / den).mean())

    p_ref, p_q = _softmax(ref_logits), _softmax(q_logits)
    prob_l1 = float(np.abs(p_ref - p_q).sum(-1).mean())          # total-variation*2
    prob_kl = float((p_ref * (np.log(p_ref + 1e-12) - np.log(p_q + 1e-12))).sum(-1).mean())

    acc_all = float((q_pred == y_true).mean())
    acc_val = float((q_pred[val_mask] == y_true[val_mask]).mean())
    # per-class flip counts
    flip_by_class = {}
    for c in range(N_CLASSES):
        m = ref_pred == c
        if m.any():
            flip_by_class[c] = int(flips[m].sum())
    worst = max(flip_by_class, key=flip_by_class.get) if flip_by_class else -1

    return {
        "acc_all": round(acc_all, 5),
        "acc_val": round(acc_val, 5),
        "flip_pct": round(100 * flips.mean(), 3),
        "flip_pct_val": round(100 * flips[val_mask].mean(), 3),
        "logit_l1": round(logit_l1, 5),
        "logit_rmse": round(logit_rmse, 5),
        "logit_maxabs": round(logit_max, 4),
        "logit_cos": round(logit_cos, 6),
        "prob_l1": round(prob_l1, 5),
        "prob_kl": round(prob_kl, 6),
        "worst_flip_class": int(worst),
        "worst_flip_n": int(flip_by_class.get(worst, 0)),
    }
