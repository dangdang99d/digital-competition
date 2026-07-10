"""E27 — true_rank_profile split by TRUE class (user ask 2026-07-09).

For each gold-label class c: rows with true label c, bucketed by the GLOBAL MSP
deciles, stacked by the rank at which c sits in granite-LS's logit ordering.
Small-n cells are annotated (n above each bar) — read bars with tiny n loosely.

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/class_rank_profile.py
"""
import numpy as np

from src.data import ALL_CLASSES

CACHE = "analysis/cache/clpvi_tiers_granite_ls.npz"
OUT = "experiments/errorpred/figures/true_rank_profile_by_class.png"
SHORT = ["read", "grep", "lsdir", "glob", "edit", "write", "patch",
         "bash", "tests", "lint", "ask", "plan", "web", "resp"]
BG = "#fcfcfb"
SEQ = ["#184f95", "#3987e5", "#9ec5f4", "#cde2fb"]      # rank 1..4 (dark -> light)
GRAY = "#c9c9c2"                                        # rank 5+


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = np.load(CACHE)
    z, y = D["logits"], D["labels"]
    C = len(ALL_CLASSES)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    order = np.argsort(-z, 1)
    rank_true = np.argmax(order == y[:, None], 1)
    qs = np.quantile(msp, np.linspace(0, 1, 11))
    dec_id = np.clip(np.searchsorted(qs, msp, side="right") - 1, 0, 9)

    fig, axes = plt.subplots(4, 4, figsize=(16.5, 12.5), facecolor=BG,
                             sharex=True, sharey=True)
    xs = np.arange(10) + 1
    bands = [("rank 1 (correct)", 0, SEQ[0]), ("rank 2", 1, SEQ[1]),
             ("rank 3", 2, SEQ[2]), ("rank 4", 3, SEQ[3])]
    for c in range(C):
        ax = axes.ravel()[c]
        ax.set_facecolor(BG)
        rows_c = y == c
        bottom = np.zeros(10)
        for lab, k, col in bands:
            v = np.array([((rank_true[rows_c & (dec_id == d)] == k).mean()
                           if (rows_c & (dec_id == d)).sum() else 0) for d in range(10)])
            ax.bar(xs, v, bottom=bottom, color=col, width=0.75,
                   edgecolor=BG, linewidth=1.2, label=lab if c == 0 else None)
            bottom += v
        v5 = np.array([((rank_true[rows_c & (dec_id == d)] >= 4).mean()
                        if (rows_c & (dec_id == d)).sum() else 0) for d in range(10)])
        ax.bar(xs, v5, bottom=bottom, color=GRAY, width=0.75,
               edgecolor=BG, linewidth=1.2, label="rank 5+" if c == 0 else None)
        for d in range(10):
            nd = int((rows_c & (dec_id == d)).sum())
            ax.text(d + 1, 1.03, f"{nd}", ha="center", fontsize=5.6, color="#6b6b64")
        ax.set_title(f"{SHORT[c]}  (n={int(rows_c.sum())})", fontsize=10, pad=12)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color="#e8e8e4", lw=0.7)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[C:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("MSP decile (1 = least confident)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of rows", fontsize=8.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("E27 · where each TRUE class sits in granite-LS's ranking, per global MSP "
                 "decile (small gray number = rows of that class in that decile)",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    fig.savefig(OUT, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"figure -> {OUT}")


if __name__ == "__main__":
    main()
