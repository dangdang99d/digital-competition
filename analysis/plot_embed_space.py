"""Plot t-SNE projection from analysis/embed_space.py (t-SNE reads better than
UMAP for this data — reworked layout).

2x3 grid:
  top-left      all points colored by confusion group (blue/green/yellow/red)
  top-middle    legend panel
  top-right     explore focus:  other groups grey, members in the 4 colors
  bottom row    edit / execute / noncode focus panels, same treatment

Palette: the 4 hues are the validated categorical red/green/blue/yellow
(CVD-checked in this assignment order; yellow's low surface contrast is
relieved by direct median labels on every panel).

Usage: python -m analysis.plot_embed_space [--proj output/proj_hist0.npz]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.data import ACTION_GROUPS, ALL_CLASSES, GROUP_ID

# user-chosen hues, validated order: blue, green, yellow, red
FOUR = ["#2a78d6", "#008300", "#eda100", "#e34948"]
GROUP_COLOR = dict(zip(ACTION_GROUPS, FOUR))          # explore/edit/execute/noncode
GREY = "#cfcec9"
INK = "#0b0b0b"


def panel(ax, title):
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d8d7d2"), s.set_linewidth(0.8)


def halo_label(ax, x, y, text, color=INK, size=10):
    ax.text(x, y, text, fontsize=size, fontweight="bold", color=color,
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default="output/proj_hist0.npz")
    ap.add_argument("--out", default="figures/tsne_hist0.png")
    ap.add_argument("--title", default="hist0 (bge-m3) [CLS] embeddings — 70k train samples, t-SNE")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    d = np.load(args.proj)
    xy, y = d["tsne"], d["y"]
    gid = np.array(GROUP_ID)
    g_of_pt = gid[y]

    fig, axes = plt.subplots(2, 3, figsize=(19, 12.5), facecolor="#fcfcfb")

    # --- top-left: all groups ---
    ax = axes[0, 0]
    colors = np.array(FOUR)[g_of_pt]
    ax.scatter(xy[:, 0], xy[:, 1], s=1.0, c=colors, alpha=0.35, linewidths=0,
               rasterized=True)
    for gi, g in enumerate(ACTION_GROUPS):
        m = g_of_pt == gi
        halo_label(ax, np.median(xy[m, 0]), np.median(xy[m, 1]), g)
    panel(ax, "t-SNE — all 4 confusion groups")

    # --- top-middle: legend panel ---
    ax = axes[0, 1]
    ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", color=GROUP_COLOR[g], markersize=10,
                      label=f"{g}:  " + ", ".join(ACTION_GROUPS[g]))
               for g in ACTION_GROUPS]
    handles.append(Line2D([], [], marker="o", ls="", color=GREY, markersize=10,
                          label="other groups (focus panels)"))
    ax.legend(handles=handles, loc="center", fontsize=10, frameon=False)
    ax.set_title(args.title, fontsize=12, color=INK)

    # --- focus panels: one per group ---
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    for (r, c), (gi, g) in zip(positions, enumerate(ACTION_GROUPS)):
        ax = axes[r, c]
        other = g_of_pt != gi
        ax.scatter(xy[other, 0], xy[other, 1], s=0.8, c=GREY, alpha=0.25,
                   linewidths=0, rasterized=True)
        members = ACTION_GROUPS[g]
        for ci, cls in enumerate(members):
            m = y == ALL_CLASSES.index(cls)
            ax.scatter(xy[m, 0], xy[m, 1], s=1.4, c=FOUR[ci], alpha=0.55,
                       linewidths=0, rasterized=True)
        for ci, cls in enumerate(members):   # labels above all scatter layers
            m = y == ALL_CLASSES.index(cls)
            halo_label(ax, np.median(xy[m, 0]), np.median(xy[m, 1]), cls,
                       color=INK, size=9)
        handles = [Line2D([], [], marker="o", ls="", color=FOUR[ci], markersize=8,
                          label=cls) for ci, cls in enumerate(members)]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        panel(ax, f"{g} — classes within group")

    fig.tight_layout()
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
