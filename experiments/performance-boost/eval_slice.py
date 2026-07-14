"""E47 helper: eval a trained checkpoint on the shared 3.5k pool slice → macro-F1
directly comparable to the anchor 0.7856 (same slice as e38_t031fd.npz). Needs
data/ + experiments/ensemble/logits/_meta.npz. Picks the best-scoring serialization
(richargs dominates, but variant-argmax matches harvest_new). Usage:
  python -m experiments.performance-boost.eval_slice --ckpt <dir> --tag e47_a1
"""
import argparse
import glob
import os

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices


def resolve(d):
    if os.path.exists(os.path.join(d, "model.safetensors")):
        return d
    cks = sorted(glob.glob(os.path.join(d, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    return d


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", default="richargs")  # comma list; harvest uses richargs,v1,richmeta
    ap.add_argument("--out", default="experiments/performance-boost/e47_slice.csv")
    args = ap.parse_args()

    samples, labels = load_samples("./data")
    yid = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, ve = train_test_split(va, test_size=0.25, stratify=yid[va], random_state=42)
    meta = np.load("experiments/ensemble/logits/_meta.npz")
    assert np.array_equal(meta["va_idx"], ve), "slice drift vs _meta.npz!"
    ytrue = yid[ve]

    ck = resolve(args.ckpt)
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ck, torch_dtype=torch.float32, trust_remote_code=True).cuda().eval()
    best = (-1.0, None)
    for v in args.variants.split(","):
        txt = build_texts([samples[i] for i in ve], variant=v)
        chunks = []
        with torch.no_grad():
            for b in range(0, len(txt), 32):
                enc = tok(txt[b:b + 32], truncation=True, max_length=512,
                          padding=True, return_tensors="pt").to("cuda")
                chunks.append(model(**enc).logits.float().cpu().numpy())
        f1 = mf1(ytrue, np.concatenate(chunks).argmax(1))
        logger.info(f"{args.tag} [{v}] slice F1={f1:.4f}")
        best = max(best, (f1, v))
    f1, v = best
    logger.success(f"{args.tag}: slice mF1={f1:.4f} ({v})  Δ vs anchor 0.7856 = {f1 - 0.7856:+.4f}")
    with open(args.out, "a") as fo:
        fo.write(f"{args.tag},{ck},{v},{f1:.4f},{f1 - 0.7856:+.4f}\n")


if __name__ == "__main__":
    main()
