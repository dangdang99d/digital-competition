"""Per-group confusion matrices for the triage LoRA specialists, one figure.

For each ft_results_lorah_<task>.csv found, maps its tag -> run dir, loads the
prune12 base + that adapter (PEFT), predicts the group's true-label val slice
(calibrated with the run's logit_bias.json), and draws a row-normalized
confusion heatmap. Groups whose adapters aren't synced yet render as "pending".

Usage: python -m analysis.lora_confusion [--out figures/lora_confusion.png]
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import ACTION_GROUPS, CLASS_TO_ID, build_texts, load_samples, split_indices

BASE = "output/pat/ft_BAAI__bge-m3_prune12/checkpoint-10500"
# sequential blue ramp (palette steps 100->700) for row-normalized shares
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, SUB = "#0b0b0b", "#52514e"


def find_runs():
    """task csv -> (group_name, classes, run_dir) via the tag/pair columns."""
    runs = {}
    for p in sorted(glob.glob("output/pat/ft_results_lorah_*.csv")):
        row = list(csv.DictReader(open(p)))[-1]
        classes = row["pair"].split(",")
        group = next(g for g, m in ACTION_GROUPS.items() if m == classes)
        run_dir = f"output/pat/ft_BAAI__bge-m3_{row['tag']}"
        ck = sorted(glob.glob(f"{run_dir}/checkpoint-*"))
        if ck:
            runs[group] = (classes, ck[-1], f"{run_dir}/logit_bias.json")
    return runs


def predict(classes, ckpt, bias_path, texts, idx, device):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(classes), ignore_mismatched_sizes=True,
        torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, ckpt).to(device).eval()
    bias = np.zeros(len(classes))
    if os.path.exists(bias_path):
        bmap = json.load(open(bias_path))["bias"]
        bias = np.array([bmap.get(c, 0.0) for c in classes])
    order = sorted(idx, key=lambda i: -len(texts[i]))
    preds = {}
    with torch.no_grad():
        for s in range(0, len(order), 32):
            chunk = order[s:s + 32]
            enc = tok([texts[i] for i in chunk], truncation=True, max_length=1024,
                      padding=True, return_tensors="pt").to(device)
            lg = model(**enc).logits.float().cpu().numpy() + bias
            for j, i in enumerate(chunk):
                preds[i] = int(lg[j].argmax())
    del model, base
    torch.cuda.empty_cache()
    return preds


def draw_panel(ax, cm, classes, title):
    n = len(classes)
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", RAMP),
              vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            c = "#ffffff" if share[i, j] > 0.55 else INK
            ax.text(j, i, f"{share[i, j]:.0%}\n{cm[i, j]:,}", ha="center", va="center",
                    fontsize=8.5, color=c, linespacing=1.4,
                    fontweight="bold" if i == j else "normal")
    short = [c.replace("_", "\n", 1) for c in classes]
    ax.set_xticks(range(n), short, fontsize=8)
    ax.set_yticks(range(n), short, fontsize=8)
    ax.set_xlabel("predicted", fontsize=8.5, color=SUB)
    ax.set_ylabel("true", fontsize=8.5, color=SUB)
    ax.set_title(title, fontsize=10.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/lora_confusion.png")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples, y = load_samples("./data")
    texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
    _, va = split_indices(y, seed=42)
    runs = find_runs()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11), facecolor="#fcfcfb")
    for ax, (g, members) in zip(axes.flat, ACTION_GROUPS.items()):
        if g not in runs:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{g}\n(adapter not synced yet)", ha="center",
                    va="center", fontsize=11, color=SUB, transform=ax.transAxes)
            continue
        classes, ckpt, bias_path = runs[g]
        idx = [i for i in va if y[i] in set(classes)]
        preds = predict(classes, ckpt, bias_path, texts, idx, device)
        n = len(classes)
        cm = np.zeros((n, n), dtype=int)
        for i in idx:
            cm[classes.index(y[i]), preds[i]] += 1
        acc = np.trace(cm) / cm.sum()
        draw_panel(ax, cm, classes, f"{g} — LoRA specialist (n={cm.sum():,}, acc {acc:.1%})")
        print(f"{g}: n={cm.sum()} acc={acc:.4f}")

    fig.suptitle("Triage LoRA specialists — within-group confusion (calibrated, val)",
                 fontsize=13, color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
