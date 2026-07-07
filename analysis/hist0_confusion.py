"""Per-group within-group confusion for the hist0 generalist — same 2x2 format
as analysis/lora_confusion.py, so the two figures overlay directly.

hist0 is one full-FT 14-class model. To compare it against the triage LoRA
specialists on the SAME decision, we restrict its 14-class logits (calibrated
with hist0's logit_bias.json) to each group's columns and argmax WITHIN the
group: "given the action is in group G, does the generalist pick the right one?"
That is exactly the task each specialist is trained on, so acc is apples-to-apples.

Usage: python -m analysis.hist0_confusion [--out figures/hist0_confusion.png]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import (ACTION_GROUPS, ALL_CLASSES, CLASS_TO_ID, build_texts,
                      load_samples, split_indices)
from analysis.lora_confusion import INK, draw_panel

HIST0 = "output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500"
BIAS_PATH = "output/pat/ft_BAAI__bge-m3_hist0/logit_bias.json"


def load_hist0(device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HIST0)
    model = AutoModelForSequenceClassification.from_pretrained(
        HIST0, num_labels=len(ALL_CLASSES), torch_dtype=torch.float16).to(device).eval()
    bias = np.zeros(len(ALL_CLASSES))
    if os.path.exists(BIAS_PATH):
        bmap = json.load(open(BIAS_PATH))["bias"]
        bias = np.array([bmap.get(c, 0.0) for c in ALL_CLASSES])
    return tok, model, bias


def predict_all(tok, model, bias, texts, idx, device):
    """14-class calibrated logits for each idx (longest-first batching)."""
    order = sorted(idx, key=lambda i: -len(texts[i]))
    logits = {}
    with torch.no_grad():
        for s in range(0, len(order), 32):
            chunk = order[s:s + 32]
            enc = tok([texts[i] for i in chunk], truncation=True, max_length=1024,
                      padding=True, return_tensors="pt").to(device)
            lg = model(**enc).logits.float().cpu().numpy() + bias
            for j, i in enumerate(chunk):
                logits[i] = lg[j]
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/hist0_confusion.png")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples, y = load_samples("./data")
    texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
    _, va = split_indices(y, seed=42)
    tok, model, bias = load_hist0(device)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11), facecolor="#fcfcfb")
    for ax, (g, classes) in zip(axes.flat, ACTION_GROUPS.items()):
        cols = [CLASS_TO_ID[c] for c in classes]          # this group's logit columns
        idx = [i for i in va if y[i] in set(classes)]
        logits = predict_all(tok, model, bias, texts, idx, device)
        n = len(classes)
        cm = np.zeros((n, n), dtype=int)
        for i in idx:
            pred_local = int(np.argmax(logits[i][cols]))   # argmax restricted to group
            cm[classes.index(y[i]), pred_local] += 1
        acc = np.trace(cm) / cm.sum()
        draw_panel(ax, cm, classes, f"{g} — hist0 generalist (n={cm.sum():,}, acc {acc:.1%})")
        print(f"{g}: n={cm.sum()} acc={acc:.4f}")

    fig.suptitle("hist0 generalist — within-group confusion, logits restricted to group "
                 "(calibrated, val)", fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
