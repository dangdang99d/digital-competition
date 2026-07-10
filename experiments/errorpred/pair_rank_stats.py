"""E27 — P(true = rank-1) vs P(true = rank-2) conditioned on the observable triple
(rank-1 class, rank-2 class, MSP decile).  User ask 2026-07-10: e.g. "rank1=read,
rank2=grep, decile d — is the truth more likely read or grep?"

Outputs:
  figures/pair_flip_heatmap.png     14x14 (r1 x r2): P(true=r2) - P(true=r1), low/high-MSP
                                    halves separately; masked where n < 30
  figures/pair_decile_profiles.png  top-16 (r1,r2) pairs by count: per-decile stacked
                                    P(true=r1) / P(true=r2) / P(true=neither), n labels
  stdout: all (r1, r2, decile) cells with n >= 20 where P(true=r2) >= P(true=r1)

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/pair_rank_stats.py
"""
import numpy as np

from src.data import ALL_CLASSES

CACHE = "analysis/cache/clpvi_tiers_granite_ls.npz"
FIGDIR = "experiments/errorpred/figures"
SHORT = ["read", "grep", "lsdir", "glob", "edit", "write", "patch",
         "bash", "tests", "lint", "ask", "plan", "web", "resp"]
BG = "#fcfcfb"
C1, C2, GRAY = "#184f95", "#3987e5", "#c9c9c2"   # P(true=r1) / P(true=r2) / neither


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    D = np.load(CACHE)
    z, y = D["logits"], D["labels"]
    C = len(ALL_CLASSES)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    order = np.argsort(-z, 1)
    r1, r2 = order[:, 0], order[:, 1]
    qs = np.quantile(msp, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(qs, msp, side="right") - 1, 0, 9)

    # ---- notable cells table ----
    print("## E27 · (rank1, rank2, decile) cells where P(true=rank2) >= P(true=rank1), n>=20\n")
    print("| rank1 | rank2 | decile | n | P(true=r1) | P(true=r2) | P(neither) |")
    print("|---|---|---|---|---|---|---|")
    for a in range(C):
        for b in range(C):
            pair = (r1 == a) & (r2 == b)
            if pair.sum() < 20:
                continue
            for d in range(10):
                m = pair & (dec == d)
                if m.sum() < 20:
                    continue
                p1, p2 = (y[m] == a).mean(), (y[m] == b).mean()
                if p2 >= p1:
                    print(f"| {SHORT[a]} | {SHORT[b]} | {d+1} | {m.sum()} | "
                          f"{p1:.3f} | {p2:.3f} | {1 - p1 - p2:.3f} |")

    # ---- fig 1: flip heatmap, low-MSP half vs high-MSP half ----
    div = LinearSegmentedColormap.from_list(
        "div", ["#e34948", "#f0a5a4", "#e8e8e4", "#86b6ef", "#2a78d6"])
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), facecolor=BG)
    for ax, (dmask, ttl) in zip(axes, [(dec <= 4, "deciles 1–5 (low MSP)"),
                                       (dec >= 5, "deciles 6–10 (high MSP)")]):
        M = np.full((C, C), np.nan)
        for a in range(C):
            for b in range(C):
                m = (r1 == a) & (r2 == b) & dmask
                if m.sum() >= 30:
                    M[a, b] = (y[m] == b).mean() - (y[m] == a).mean()
        im = ax.imshow(M, cmap=div, vmin=-1, vmax=1, aspect="equal")
        for a in range(C):
            for b in range(C):
                if not np.isnan(M[a, b]):
                    n = int(((r1 == a) & (r2 == b) & dmask).sum())
                    ax.text(b, a, f"{M[a, b]:+.2f}\n{n}", ha="center", va="center",
                            fontsize=5.4, color="#1a1a19")
        for bnd in (3.5, 6.5, 9.5):
            ax.axhline(bnd, color="#9a9a92", lw=0.7)
            ax.axvline(bnd, color="#9a9a92", lw=0.7)
        ax.set_xticks(range(C), SHORT, fontsize=7, rotation=90)
        ax.set_yticks(range(C), SHORT, fontsize=7)
        ax.tick_params(length=0)
        ax.set_xlabel("rank-2 class")
        ax.set_title(ttl, fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("rank-1 class (prediction)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01,
                 label="P(true = rank2) − P(true = rank1)   (blue = rank1 safe, red = flip)")
    fig.suptitle("E27 · does the runner-up ever beat the prediction? — by (rank1, rank2) pair "
                 "(cell text: Δ and n; blank = n<30)", fontsize=12.5, y=0.98)
    fig.savefig(f"{FIGDIR}/pair_flip_heatmap.png", dpi=130, facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)

    # ---- fig 2: per-decile profiles for the 16 most frequent pairs ----
    counts = np.zeros((C, C), int)
    np.add.at(counts, (r1, r2), 1)
    top = np.dstack(np.unravel_index(np.argsort(counts.ravel())[::-1][:16], (C, C)))[0]
    fig, axes = plt.subplots(4, 4, figsize=(16.5, 12), facecolor=BG,
                             sharex=True, sharey=True)
    xs = np.arange(10) + 1
    for (a, b), ax in zip(top, axes.ravel()):
        ax.set_facecolor(BG)
        pair = (r1 == a) & (r2 == b)
        v1 = np.array([((y[pair & (dec == d)] == a).mean()
                        if (pair & (dec == d)).sum() else 0) for d in range(10)])
        v2 = np.array([((y[pair & (dec == d)] == b).mean()
                        if (pair & (dec == d)).sum() else 0) for d in range(10)])
        rest = np.array([(1 - v1[d] - v2[d]) if (pair & (dec == d)).sum() else 0
                         for d in range(10)])
        ax.bar(xs, v1, color=C1, width=0.75, edgecolor=BG, linewidth=1.2,
               label="true = rank1" if (a, b) == tuple(top[0]) else None)
        ax.bar(xs, v2, bottom=v1, color=C2, width=0.75, edgecolor=BG, linewidth=1.2,
               label="true = rank2" if (a, b) == tuple(top[0]) else None)
        ax.bar(xs, rest, bottom=v1 + v2, color=GRAY, width=0.75, edgecolor=BG,
               linewidth=1.2, label="true = neither" if (a, b) == tuple(top[0]) else None)
        for d in range(10):
            nd = int((pair & (dec == d)).sum())
            ax.text(d + 1, 1.03, f"{nd}", ha="center", fontsize=5.4, color="#6b6b64")
        ax.set_title(f"rank1 {SHORT[a]} → rank2 {SHORT[b]}   (n={int(pair.sum())})",
                     fontsize=9.5, pad=12)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color="#e8e8e4", lw=0.7)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel("MSP decile", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("share", fontsize=8.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("E27 · top-16 (rank1 → rank2) pairs: who's actually right, per MSP decile "
                 "(gray n = rows per bar)", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    fig.savefig(f"{FIGDIR}/pair_decile_profiles.png", dpi=130, facecolor=BG,
                bbox_inches="tight")
    print(f"\nfigures -> {FIGDIR}/pair_flip_heatmap.png · pair_decile_profiles.png")


if __name__ == "__main__":
    main()
