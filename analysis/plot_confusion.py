"""Wrong-prediction confusion heatmap from a val_debug dump's _summary.csv.

Sequential blue ramp (light = few errors, dark = many), sqrt-normalized so the
long tail of small counts stays readable next to the 449-error peak. Zero cells
stay at the surface color and unlabeled; nonzero cells are annotated.

Usage:
  python -m analysis.plot_confusion \
      --summary output/val_debug_bgem3/_summary.csv \
      --out figures/confusion_wrong_bgem3.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from src.data import ALL_CLASSES

# sequential blue ramp, steps 100->700 (validated palette; light surface)
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#ffffff"
INK, INK_MUTED = "#1a1a1a", "#6b6b6b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="output/val_debug_bgem3/_summary.csv")
    ap.add_argument("--out", default="figures/confusion_wrong_bgem3.png")
    ap.add_argument("--title", default="bge-m3 v1 — wrong predictions (val, n=3,570)")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    w = df[~df["correct"]]
    m = pd.crosstab(w["true"], w["pred"]).reindex(
        index=ALL_CLASSES, columns=ALL_CLASSES, fill_value=0).values.astype(float)

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUES)
    cmap.set_under(SURFACE)                      # zero cells recede to the surface
    norm = PowerNorm(gamma=0.5, vmin=1, vmax=m.max())

    fig, ax = plt.subplots(figsize=(11.5, 9.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # 2px surface gap between cells: draw via pcolormesh edges
    mesh = ax.pcolormesh(m, cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=2)

    n = len(ALL_CLASSES)
    ax.set_xticks(np.arange(n) + 0.5, ALL_CLASSES, rotation=45, ha="right",
                  fontsize=9, color=INK)
    ax.set_yticks(np.arange(n) + 0.5, ALL_CLASSES, fontsize=9, color=INK)
    ax.invert_yaxis()                            # true classes read top-to-bottom
    ax.set_xlabel("model's prediction", fontsize=10, color=INK_MUTED)
    ax.set_ylabel("correct answer", fontsize=10, color=INK_MUTED)
    ax.set_title(args.title, fontsize=12, color=INK, pad=14, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # annotate nonzero cells; ink flips to white on dark cells for contrast
    thresh = norm.inverse(0.55)                  # cells darker than ~55% get white ink
    for i in range(n):
        for j in range(n):
            v = int(m[i, j])
            if v == 0:
                continue
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if v >= thresh else INK)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.02, extend="min")
    cbar.set_label("number of wrong predictions", fontsize=9, color=INK_MUTED)
    cbar.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_MUTED)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
