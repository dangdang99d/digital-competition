"""E27 — rank profile grouped by PREDICTED class × MSP quantile bin (inference-accessible).

Conditions only on observables: for rows whose rank-1 PREDICTION is class c, in MSP
quantile-bin b (global, ascending), where does the true label sit in the ranking?
(rank-1 share = accuracy of that cell; rank-2 share = how often the runner-up is right.)

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/pred_class_rank_profile.py [--bins 20]
"""
import argparse

import numpy as np

from src.data import ALL_CLASSES

CACHE = "analysis/cache/clpvi_tiers_granite_ls.npz"
SHORT = ["read", "grep", "lsdir", "glob", "edit", "write", "patch",
         "bash", "tests", "lint", "ask", "plan", "web", "resp"]
BG = "#fcfcfb"
SEQ = ["#184f95", "#3987e5", "#9ec5f4", "#cde2fb"]      # truth at rank 1..4
GRAY = "#c9c9c2"                                        # truth at rank 5+


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    B = args.bins

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = np.load(CACHE)
    z, y = D["logits"], D["labels"]
    C = len(ALL_CLASSES)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    order = np.argsort(-z, 1)
    pred = order[:, 0]
    rank_true = np.argmax(order == y[:, None], 1)
    qs = np.quantile(msp, np.linspace(0, 1, B + 1))
    bin_id = np.clip(np.searchsorted(qs, msp, side="right") - 1, 0, B - 1)

    fig, axes = plt.subplots(4, 4, figsize=(1.05 * B + 6.5, 12.5), facecolor=BG,
                             sharex=True, sharey=True)
    xs = np.arange(B) + 1
    bands = [("rank 1 = prediction correct", 0, SEQ[0]), ("true = rank 2", 1, SEQ[1]),
             ("true = rank 3", 2, SEQ[2]), ("true = rank 4", 3, SEQ[3])]
    for c in range(C):
        ax = axes.ravel()[c]
        ax.set_facecolor(BG)
        rows_c = pred == c                                # observable at inference
        bottom = np.zeros(B)
        for lab, k, col in bands:
            v = np.array([((rank_true[rows_c & (bin_id == b)] == k).mean()
                           if (rows_c & (bin_id == b)).sum() else 0) for b in range(B)])
            ax.bar(xs, v, bottom=bottom, color=col, width=0.75,
                   edgecolor=BG, linewidth=0.8, label=lab if c == 0 else None)
            bottom += v
        v5 = np.array([((rank_true[rows_c & (bin_id == b)] >= 4).mean()
                        if (rows_c & (bin_id == b)).sum() else 0) for b in range(B)])
        ax.bar(xs, v5, bottom=bottom, color=GRAY, width=0.75,
               edgecolor=BG, linewidth=0.8, label="true = rank 5+" if c == 0 else None)
        for b in range(B):
            nb = int((rows_c & (bin_id == b)).sum())
            ax.text(b + 1, 1.03, f"{nb}", ha="center", fontsize=4.6, color="#6b6b64",
                    rotation=90 if B > 12 else 0, va="bottom")
        ax.set_title(f"predicted {SHORT[c]}  (n={int(rows_c.sum())})", fontsize=10, pad=16)
        ax.set_ylim(0, 1.16)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks(xs[1::2] if B > 12 else xs)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color="#e8e8e4", lw=0.7)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[C:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(f"MSP {100 // B}%-bin (1 = least confident)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of rows", fontsize=8.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"E27 · rows grouped by PREDICTED class (inference-accessible) × MSP "
                 f"{B}-quantile bin — where the true label sits in granite-LS's ranking "
                 f"(gray n = rows per bar)", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    out = (f"experiments/errorpred/figures/true_rank_profile_by_predclass"
           + (f"_b{B}" if B != 10 else "") + ".png")
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
