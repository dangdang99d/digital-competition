"""plot_drop_maps.py — datamap.png-style cartography maps, one per keep-set method,
with that method's DROPPED train samples overlaid (user ask 2026-07-09).

Same axes as figures/datamap.png (x = variability of gold-label prob over epochs,
y = mean confidence), from the instrumented baseline's dynamics npz. Each panel:
full 56k train cloud in gray + the method's dropped samples in red. Shows WHERE each
scorer cuts — the §2 winners (pvi/aum) should sit in the hard/mislabeled bottom-left,
the losers (cart/el2n) eat into ambiguous/easy mass.

CPU-only:  PYTHONPATH=. python experiments/coreset/plot_drop_maps.py
"""
import numpy as np

DYN = "analysis/cache/coreset_dyn_granite.npz"
OOF = "analysis/cache/coreset_oof_granite.npz"
KEEPDIR = "experiments/coreset/keepsets"
FIGDIR = "experiments/coreset/figures"
BG = "#fcfcfb"
# (keepset file, label, Δ vs base from eval_results.md)
PANELS = [
    ("pvi_drop06.npy",           "pvi06 · drop 6%",   "+0.0059"),
    ("aum_drop06.npy",           "aum06 · drop 6%",   "+0.0047"),
    ("cleanlab_keep.npy",        "cl · drop 20.2%",   "+0.0001"),
    ("forget_neverlearned.npy",  "forget · drop 10.1%", "+0.0009"),
    ("el2n_drop06.npy",          "el2n06 · drop 6%",  "−0.0045"),
    ("cart_drophard_drop06.npy", "cart06 · drop 6%",  "−0.0065"),
]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = np.load(DYN)
    P, tr, y = D["probs"], D["tr"], D["labels"]        # P:(E,N,C) over the 56k train
    idx = np.arange(P.shape[1])
    ptrue = P[:, idx, y]
    conf, var = ptrue.mean(0), ptrue.std(0)
    pos = {a: i for i, a in enumerate(tr)}             # absolute idx -> row in dynamics

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), sharex=True, sharey=True,
                             facecolor=BG)
    for ax, (kf, label, d) in zip(axes.ravel(), PANELS):
        keep = np.load(f"{KEEPDIR}/{kf}")
        dropped = np.setdiff1d(tr, keep)               # absolute indices dropped
        rows = np.array([pos[a] for a in dropped])
        ax.set_facecolor(BG)
        ax.scatter(var, conf, color="#c9c9c2", s=3, alpha=0.25, rasterized=True)
        ax.scatter(var[rows], conf[rows], color="#e34948", s=4, alpha=0.5,
                   rasterized=True)
        ax.set_title(f"{label}  (ΔF1 {d})", fontsize=11)
        ax.annotate(f"dropped n={len(rows)}\nmean conf {conf[rows].mean():.2f} · "
                    f"mean var {var[rows].mean():.2f}",
                    (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                    fontsize=8.5, color="#6b6b64")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(color="#e8e8e4", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("variability (std of gold-label prob)")
    for ax in axes[:, 0]:
        ax.set_ylabel("confidence (mean gold-label prob)")
    fig.suptitle("What each drop-noisy method removes — dropped samples (red) on the data map\n"
                 "easy top-left · ambiguous right · hard/mislabeled bottom-left | gray = all 56k train",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{FIGDIR}/datamap_drops.png", dpi=130, facecolor=BG)
    print(f"saved {FIGDIR}/datamap_drops.png")

    # ---- PVI variant of the base map: same axes, colored by PVI (the winning score) ----
    O = np.load(OOF)
    oof, tr_o, y_o = O["oof"], O["tr"], O["labels"]
    assert np.array_equal(tr_o, tr)
    prior = np.bincount(y_o, minlength=oof.shape[1]) / len(tr_o)
    pvi = np.log2(np.clip(oof[np.arange(len(tr_o)), y_o], 1e-6, 1)) - np.log2(prior[y_o] + 1e-12)
    fig, ax = plt.subplots(figsize=(7.4, 5.8), facecolor=BG)
    ax.set_facecolor(BG)
    lim = np.percentile(np.abs(pvi), 98)
    sc = ax.scatter(var, conf, c=pvi, cmap="RdYlBu", vmin=-lim, vmax=lim, s=4,
                    alpha=0.5, rasterized=True)
    ax.set_xlabel("variability (std of gold-label prob)")
    ax.set_ylabel("confidence (mean gold-label prob)")
    ax.set_title("Data map colored by PVI — blue = informative, red = unlearnable/mislabeled\n"
                 "(pvi06 drops the reddest 6%)", fontsize=11.5)
    fig.colorbar(sc, label="PVI (bits)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color="#e8e8e4", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/datamap_pvi.png", dpi=130, facecolor=BG)
    print(f"saved {FIGDIR}/datamap_pvi.png")


if __name__ == "__main__":
    main()
