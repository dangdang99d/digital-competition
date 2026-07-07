"""Plot the FFN (and KV overlay) whitened-SVD degradation curve from palu_probe_ffn JSON."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, SUB = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, RED = "#3987e5", "#d4553f"
GRID = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.json))
    base = d["baseline"]
    curves = d["curves"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=BG)
    style = {"ffn": (BLUE, "o", "FFN (gate/up/down)"),
             "kv":  (RED, "s", "K/V (attn, ref)")}

    for ax, xkey, xlabel in ((axL, "ratio", "rank-keep ratio  (r / min(out,in))"),
                             (axR, "param", "params kept per matrix  (factorized)")):
        ax.set_facecolor(BG)
        ax.axhline(base, color=SUB, lw=1.0, ls="--", zorder=1)
        ax.text(0.015, base, f"baseline {base:.4f}", color=SUB, fontsize=8.5,
                va="bottom", ha="left", transform=ax.get_yaxis_transform())
        for name, rows in curves.items():
            col, mk, lab = style[name]
            xs = [r[xkey] for r in rows]
            ys = [r["f1"] for r in rows]
            ax.plot(xs, ys, color=col, marker=mk, ms=5.5, lw=1.8, label=lab, zorder=3)
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=SUB, labelsize=9)
        ax.invert_xaxis()

    axL.set_ylabel("uncalibrated macro-F1 (3k val)", color=INK, fontsize=10)
    axL.legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.suptitle("Whitened-SVD low-rank degradation — champion Qwen3-0.6B (training-free)",
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150, facecolor=BG)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
