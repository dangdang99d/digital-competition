"""First-step (zero-history) failure plots for hist0 (bge-m3) and qwen3.

Per model, two figures on the SAME 1807 first-step val slice (seed 42):
  figures/firststep_acc_<model>.png  — per-class accuracy bars (n annotated)
  figures/firststep_conf_<model>.png — 14x14 row-normalized confusion heatmap
Same class order in all four so the two models compare cell-for-cell.
UNCALIBRATED: preds = argmax of the raw cached logits (analysis/cache/*_val_logits.npz).
The logit bias is fit on this val set, so error analysis must not look through it.
"""
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, split_indices

# sequential blue ramp (same as figures/lora_confusion.png) + ink tokens
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BAR = "#3987e5"
INK, SUB, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
BG = "#fcfcfb"
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("b", RAMP)

MODELS = {"hist0": "hist0 (bge-m3 full-FT)", "qwen3": "Qwen3-0.6B full-FT"}


def acc_plot(name, title, y_true, preds, order):
    accs, ns = [], []
    for c in order:
        m = y_true == c
        ns.append(int(m.sum()))
        accs.append((preds[m] == c).mean() if m.sum() else np.nan)
    fig, ax = plt.subplots(figsize=(8.6, 6.4), facecolor=BG)
    ax.set_facecolor(BG)
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, accs, height=0.62, color=BAR, zorder=3)
    for yp, a, n in zip(ypos, accs, ns):
        ax.text(a + 0.012, yp, f"{a:.0%}", va="center", fontsize=8.5, color=INK)
        ax.text(1.13, yp, f"n={n}", va="center", ha="right", fontsize=8, color=SUB)
    ax.set_yticks(ypos, [ALL_CLASSES[c] for c in order], fontsize=9)
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, .25, .5, .75, 1.0], ["0%", "25%", "50%", "75%", "100%"],
                  fontsize=8.5, color=SUB)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ov_acc = (preds == y_true).mean()
    mf1 = f1_score(y_true, preds, average="macro")
    ax.set_title(f"{title}\nfirst-step val (n={len(y_true):,}) — acc {ov_acc:.1%}, "
                 f"macro-F1 {mf1:.4f}", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("per-class accuracy on zero-history samples", fontsize=9, color=SUB)
    fig.tight_layout()
    out = f"figures/firststep_acc_{name}.png"
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)
    print("saved ->", out)


def conf_plot(name, title, y_true, preds, order):
    n = len(order)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, preds):
        cm[order.index(t), order.index(p)] += 1
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10.6, 9.6), facecolor=BG)
    ax.imshow(share, cmap=CMAP, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if share[i, j] < 0.02:          # annotate only visible cells
                continue
            c = "#ffffff" if share[i, j] > 0.55 else INK
            ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=7.5, color=c,
                    fontweight="bold" if i == j else "normal")
    labels = [ALL_CLASSES[c] for c in order]
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("predicted", fontsize=9, color=SUB)
    ax.set_ylabel("true", fontsize=9, color=SUB)
    ax.set_title(f"{title} — first-step confusion (row-normalized, uncalibrated, "
                 f"n={len(y_true):,})", fontsize=11, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    out = f"figures/firststep_conf_{name}.png"
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)
    print("saved ->", out)


def main():
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    y_all = np.array([CLASS_TO_ID[y[i]] for i in va])
    step = np.array([int(samples[i]["id"].split("step_")[1]) for i in va])
    first = step == 1
    y_true = y_all[first]

    # one shared class order: first-step true-label frequency, descending
    freq = collections.Counter(y_true)
    order = [c for c, _ in freq.most_common()] + \
            [c for c in range(len(ALL_CLASSES)) if c not in freq]

    accs = {}
    for name, title in MODELS.items():
        # uncalibrated: argmax of raw logits (no logit-bias — project decision)
        preds = np.load(f"analysis/cache/{name}_val_logits.npz")["logits"].argmax(1)[first]
        accs[name] = {c: (preds[y_true == c] == c).mean() for c in order if (y_true == c).sum()}
        acc_plot(name, title, y_true, preds, order)
        conf_plot(name, title, y_true, preds, order)

    print("\nper-class accuracy difference (qwen3 - hist0), first-step:")
    for c in order:
        if c in accs["hist0"]:
            d = accs["qwen3"][c] - accs["hist0"][c]
            print(f"  {ALL_CLASSES[c]:20} {d:+.1%}")


if __name__ == "__main__":
    main()
