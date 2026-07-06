"""Plot UMAP/t-SNE projections from analysis/embed_space.py.

2x2 grid: rows = UMAP / t-SNE, cols = colored by confusion group / by 14 classes.
Group colors: 4 validated categorical hues (CVD-checked). Class colors: composite
encoding — the group's hue, stepped in lightness per member (14 unrelated hues
would be unreadable and CVD-unsafe at point scale). Direct group labels at
cluster medians satisfy the contrast relief rule for the aqua/yellow slots.

Usage: python -m analysis.plot_embed_space [--proj output/proj_hist0.npz]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import ACTION_GROUPS, ALL_CLASSES, GROUP_ID

GROUP_HEX = {"explore": "#2a78d6", "edit": "#1baf7a",
             "execute": "#eda100", "noncode": "#008300"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e"}


def shade(hex_color, f):
    """f in [-1, 1]: negative blends toward white, positive toward black."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    target = np.zeros(3) if f > 0 else np.full(3, 255.0)
    out = rgb + (target - rgb) * abs(f)
    return "#" + "".join(f"{int(round(v)):02x}" for v in out)


def class_colors():
    """Per-class color: group hue stepped in lightness (light -> dark within group)."""
    steps = {2: [-0.35, 0.25], 3: [-0.4, 0.0, 0.35], 4: [-0.45, -0.15, 0.15, 0.45]}
    colors = {}
    for g, members in ACTION_GROUPS.items():
        for c, f in zip(members, steps[len(members)]):
            colors[c] = shade(GROUP_HEX[g], f)
    return [colors[c] for c in ALL_CLASSES]


def scatter_panel(ax, xy, colors_per_point, title):
    ax.scatter(xy[:, 0], xy[:, 1], s=1.0, c=colors_per_point, alpha=0.35,
               linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=11, color=INK["primary"])
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d8d7d2"), s.set_linewidth(0.8)


def label_groups(ax, xy, y, gid):
    for gi, g in enumerate(ACTION_GROUPS):
        m = gid[y] == gi
        cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
        ax.text(cx, cy, g, fontsize=10, fontweight="bold", color=INK["primary"],
                ha="center", va="center",
                path_effects=[__import__("matplotlib.patheffects", fromlist=["w"])
                              .withStroke(linewidth=2.5, foreground="white")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default="output/proj_hist0.npz")
    ap.add_argument("--out", default="figures/embed_space_hist0.png")
    args = ap.parse_args()

    d = np.load(args.proj)
    y = d["y"]
    gid = np.array(GROUP_ID)
    gcolors = np.array([GROUP_HEX[g] for g in ACTION_GROUPS])[gid[y]]
    ccolors = np.array(class_colors())[y]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), facecolor="#fcfcfb")
    for row, (name, key) in enumerate([("UMAP", "umap"), ("t-SNE", "tsne")]):
        xy = d[key]
        scatter_panel(axes[row, 0], xy, gcolors, f"{name} — by confusion group")
        label_groups(axes[row, 0], xy, y, gid)
        scatter_panel(axes[row, 1], xy, ccolors, f"{name} — by action (shade = member)")

    # legends: groups (left col) and grouped 14-class (right col)
    from matplotlib.lines import Line2D
    gh = [Line2D([], [], marker="o", ls="", color=GROUP_HEX[g], markersize=8, label=g)
          for g in ACTION_GROUPS]
    ch = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=a)
          for a, c in zip(ALL_CLASSES, class_colors())]
    axes[0, 0].legend(handles=gh, loc="upper right", fontsize=8, framealpha=0.9)
    axes[0, 1].legend(handles=ch, loc="upper right", fontsize=6.5, framealpha=0.9, ncol=2)
    fig.suptitle("hist0 (bge-m3) [CLS] embedding space — 70k train samples",
                 fontsize=13, color=INK["primary"], y=0.995)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
