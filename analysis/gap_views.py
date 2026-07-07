"""More uncalibrated logit-gap views from the cached val logits.

Per model (hist0, qwen3):
  figures/logit_gap_first_<model>.png    — gap dist, correct vs wrong, ZERO-HISTORY val only
  figures/logit_gap_byclass_<model>.png  — per true class: median gap + IQR, correct vs wrong
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, split_indices

MODELS = {"hist0": "hist0 (bge-m3 full-FT)", "qwen3": "Qwen3-0.6B full-FT"}
BLUE, RED = "#3987e5", "#d4553f"
INK, SUB, GRID, BG = "#0b0b0b", "#52514e", "#e4e2dd", "#fcfcfb"


def first_fig(name, title, gap, correct, first):
    g, c = gap[first], correct[first]
    xmax = np.percentile(gap, 99.5)
    bins = np.linspace(0, xmax, 40)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), facecolor=BG, sharey=True)
    for ax, mask, color, label in ((axes[0], c, BLUE, "correct"),
                                   (axes[1], ~c, RED, "incorrect")):
        ax.set_facecolor(BG)
        ax.hist(g[mask], bins=bins, color=color, density=True, zorder=3)
        med = np.median(g[mask])
        ax.axvline(med, color=INK, lw=1.2, ls="--", zorder=4)
        ax.text(med + xmax * .015, 0.9, f"median {med:.2f}",
                transform=ax.get_xaxis_transform(), fontsize=8.5, color=INK)
        ax.set_title(f"{label} (n={mask.sum():,}, {mask.mean():.1%} of first-step)",
                     fontsize=10.5, color=INK)
        ax.set_xlabel("top-1 − top-2 logit gap (uncalibrated)", fontsize=9, color=SUB)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(labelsize=8.5, colors=SUB)
    axes[0].set_ylabel("density", fontsize=9, color=SUB)
    fig.suptitle(f"{title} — decision margin on ZERO-HISTORY val (n={first.sum():,})",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = f"figures/logit_gap_first_{name}.png"
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)
    print("saved ->", out)


def byclass_fig(name, title, gap, correct, y_true):
    # order classes by full-val frequency, most common on top
    counts = np.bincount(y_true, minlength=len(ALL_CLASSES))
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(9.4, 7.6), facecolor=BG)
    ax.set_facecolor(BG)
    OFF = 0.18
    for row, cls in enumerate(order):
        yc = len(order) - 1 - row
        for mask, color, off in ((correct, BLUE, +OFF), (~correct, RED, -OFF)):
            g = gap[(y_true == cls) & mask]
            if len(g) < 5:
                continue
            q1, med, q3 = np.percentile(g, [25, 50, 75])
            ax.plot([q1, q3], [yc + off] * 2, color=color, lw=2.4,
                    solid_capstyle="round", zorder=3, alpha=0.85)
            ax.plot(med, yc + off, "o", color=color, ms=5.5, zorder=4,
                    mec=BG, mew=1.0)
        n_w = int(((y_true == cls) & ~correct).sum())
        ax.text(ax.get_xlim()[1], yc, "", fontsize=7)  # placeholder keeps xlim stable
        ax.text(1.005, yc / len(order) + 0.5 / len(order) - 0.5 / len(order),
                "", transform=ax.get_yaxis_transform())
    ax.set_yticks([len(order) - 1 - r for r in range(len(order))],
                  [f"{ALL_CLASSES[c]}  (err n={int(((y_true==c)&~correct).sum()):,})"
                   for c in order], fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(labelsize=8.5, colors=SUB)
    ax.set_xlabel("top-1 − top-2 logit gap (uncalibrated) — median dot, IQR bar",
                  fontsize=9, color=SUB)
    ax.plot([], [], "o-", color=BLUE, label="correct")
    ax.plot([], [], "o-", color=RED, label="incorrect")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title(f"{title} — decision margin by true class (full val)",
                 fontsize=11.5, color=INK, loc="left")
    fig.tight_layout()
    out = f"figures/logit_gap_byclass_{name}.png"
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)
    print("saved ->", out)


def main():
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])
    first = np.array([len(samples[i]["history"]) == 0 for i in va])
    for name, title in MODELS.items():
        lg = np.load(f"analysis/cache/{name}_val_logits.npz")["logits"]
        top2 = np.sort(lg, axis=1)[:, -2:]
        gap = top2[:, 1] - top2[:, 0]
        correct = lg.argmax(1) == y_true
        first_fig(name, title, gap, correct, first)
        byclass_fig(name, title, gap, correct, y_true)


if __name__ == "__main__":
    main()
