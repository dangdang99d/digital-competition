"""E27 — per-MSP-decile confusion / rank structure of granite-LS val errors.

User ask 2026-07-09: for each confidence decile, plot the confusion matrix for the
model's top-1 prediction AND for the counterfactual "pick the 2nd/3rd/4th-best class"
— hunting for structure that hints at the true label when top-1 is likely wrong
(known anchors: §5 group acc ~0.99; E6 top-2 fallback net-negative).

Data: cached granite-LS e9 val logits (analysis/cache/clpvi_tiers_granite_ls.npz,
standard split — val out-of-sample). Raw fp32 logits, NO calibration. CPU-only.

Outputs:
  figures/rank{1..4}_confusion_by_decile.png  (row-normalized 14x14 per decile)
  figures/true_rank_profile.png               (where the true label sits, per decile)
  markdown stat tables on stdout (per-decile acc@1, P(true=rank k | wrong), top-k, group)

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/decile_confusion.py
"""
import os

import numpy as np

from src.data import ALL_CLASSES, GROUP_ID

CACHE = "analysis/cache/clpvi_tiers_granite_ls.npz"
FIGDIR = "experiments/errorpred/figures"
SHORT = ["read", "grep", "lsdir", "glob", "edit", "write", "patch",
         "bash", "tests", "lint", "ask", "plan", "web", "resp"]
BG = "#fcfcfb"
# sequential blue ramp (dataviz reference palette, steps 100->700)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    D = np.load(CACHE)
    z, y = D["logits"], D["labels"]
    C = len(ALL_CLASSES)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    order = np.argsort(-z, 1)                       # order[:,k] = (k+1)-th best class
    rank_true = np.argmax(order == y[:, None], 1)   # 0-based rank of the gold label
    gid = np.array(GROUP_ID)

    n = len(y)
    dec = np.argsort(msp)                           # ascending: least confident first
    deciles = [dec[d * n // 10:(d + 1) * n // 10] for d in range(10)]

    # ---- stat tables ----
    print("## E27 · per-decile rank structure (granite-LS e9, 14k val, raw)\n")
    print("| decile (MSP asc) | MSP range | acc@1 | P(true=r2|wrong) | P(true=r3|wrong) | "
          "P(true∈top2) | top3 | top4 | P(true in r1 group) | P(true in r1 group|wrong) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d, idx in enumerate(deciles):
        rt = rank_true[idx]
        wrong = rt != 0
        in_grp = gid[order[idx, 0]] == gid[y[idx]]
        print(f"| {d+1} | {msp[idx].min():.2f}–{msp[idx].max():.2f} | {(~wrong).mean():.3f} | "
              f"{(rt[wrong] == 1).mean():.3f} | {(rt[wrong] == 2).mean():.3f} | "
              f"{(rt < 2).mean():.3f} | {(rt < 3).mean():.3f} | {(rt < 4).mean():.3f} | "
              f"{in_grp.mean():.3f} | {in_grp[wrong].mean():.3f} |")

    print("\n| decile | P(r2 in r1 group) | P(true=r2 | wrong & r2 in grp) | "
          "P(true=r2 | wrong & r2 out grp) |")
    print("|---|---|---|---|")
    for d, idx in enumerate(deciles):
        rt, o = rank_true[idx], order[idx]
        wrong = rt != 0
        r2grp = gid[o[:, 1]] == gid[o[:, 0]]
        a = rt[wrong & r2grp] == 1
        b = rt[wrong & ~r2grp] == 1
        print(f"| {d+1} | {r2grp.mean():.3f} | "
              f"{a.mean() if len(a) else float('nan'):.3f} | "
              f"{b.mean() if len(b) else float('nan'):.3f} |")

    _figures(deciles, order, y, msp, rank_true)


def _figures(deciles, order, y, msp, rank_true):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", [BG] + SEQ)
    C = len(ALL_CLASSES)

    # --- rank-k confusion per decile (row-normalized over true class) ---
    for k in range(4):
        fig, axes = plt.subplots(2, 5, figsize=(16, 7.4), facecolor=BG)
        for d, ax in enumerate(axes.ravel()):
            idx = deciles[d]
            M = np.zeros((C, C))
            np.add.at(M, (y[idx], order[idx, k]), 1)
            M = M / np.maximum(M.sum(1, keepdims=True), 1)
            ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="equal")
            for b in (3.5, 6.5, 9.5):               # group blocks: explore|edit|execute|noncode
                ax.axhline(b, color="#9a9a92", lw=0.7)
                ax.axvline(b, color="#9a9a92", lw=0.7)
            ax.set_title(f"decile {d+1}  (MSP {msp[idx].min():.2f}–{msp[idx].max():.2f})",
                         fontsize=8.5)
            ax.set_xticks(range(C), SHORT, fontsize=5.5, rotation=90)
            ax.set_yticks(range(C), SHORT, fontsize=5.5)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.suptitle(f"E27 · granite-LS val — P(rank-{k+1} class = col | true = row) per MSP "
                     f"decile (1 = least confident); gray lines = action-group blocks",
                     fontsize=12, y=0.99)
        sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_clim(0, 1)
        fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
        fig.savefig(f"{FIGDIR}/rank{k+1}_confusion_by_decile.png", dpi=130,
                    facecolor=BG, bbox_inches="tight")
        plt.close(fig)

    # --- where does the true label sit, per decile ---
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    xs = np.arange(10) + 1
    bands = [("rank 1 (correct)", rank_true == 0, SEQ[5]),
             ("rank 2", rank_true == 1, SEQ[3]),
             ("rank 3", rank_true == 2, SEQ[1]),
             ("rank 4", rank_true == 3, SEQ[0]),
             ("rank 5+", rank_true >= 4, "#c9c9c2")]
    bottom = np.zeros(10)
    for lab, mask, col in bands:
        v = np.array([mask[idx].mean() for idx in deciles])
        ax.bar(xs, v, bottom=bottom, color=col, width=0.72,
               edgecolor=BG, linewidth=2, label=lab)
        for x, (b, h) in enumerate(zip(bottom, v)):
            if h > 0.045:
                ax.text(x + 1, b + h / 2, f"{h:.0%}", ha="center", va="center",
                        fontsize=7.5, color="#1a1a19" if col != SEQ[5] else "#ffffff")
        bottom += v
    ax.set_xticks(xs, [f"{d}" for d in xs])
    ax.set_xlabel("MSP decile (1 = least confident)")
    ax.set_ylabel("share of rows")
    ax.set_title("Where the true label sits in granite-LS's ranking, per confidence decile")
    ax.legend(frameon=False, fontsize=8.5, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#e8e8e4", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/true_rank_profile.png", dpi=130, facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigures -> {FIGDIR}/rank{{1..4}}_confusion_by_decile.png · true_rank_profile.png")


if __name__ == "__main__":
    main()
