"""Visualize the within-group discriminators (see analysis/group_discriminators.py).

One figure per group, two panels:
  left  — heatmap: distribution of last history action within each class
          (columns sum to 100%; sequential blue = share)
  right — grouped bars: structural metadata rates per class
          (turn=first, git dirty, files open, ci=failed)

Usage: python -m analysis.plot_discriminators   # writes figures/discriminators_<group>.png
"""
import json
import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

GROUPS = {
    "explore": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "non-code": ["ask_user", "plan_task", "web_search"],
}
# validated palette: sequential blue ramp + categorical slots 1..4 (light mode)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
SURFACE, INK, INK_MUTED = "#ffffff", "#1a1a1a", "#6b6b6b"


def main():
    samples = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
    labels = {r["id"]: r["action"] for r in csv.DictReader(open("data/train_labels.csv", encoding="utf-8"))}

    for gname, classes in GROUPS.items():
        last_act = {c: Counter() for c in classes}
        meta = {c: Counter() for c in classes}
        n_cls = Counter()
        for s in samples:
            lbl = labels[s["id"]]
            if lbl not in classes:
                continue
            n_cls[lbl] += 1
            la = next((t["name"] for t in reversed(s["history"]) if t.get("role") != "user"), "<none>")
            last_act[lbl][la] += 1
            sm, ws = s["session_meta"], s["session_meta"]["workspace"]
            meta[lbl]["turn = first"] += sm["turn_index"] <= 1
            meta[lbl]["git dirty"] += bool(ws["git_dirty"])
            meta[lbl]["files open"] += len(ws["open_files"]) > 0
            meta[lbl]["ci = failed"] += ws["last_ci_status"] == "failed"

        # top last-actions by overall mass + an explicit remainder row, so every
        # column sums to exactly 100%
        overall = Counter()
        for c in classes:
            overall.update(last_act[c])
        acts = [a for a, _ in overall.most_common(9)]
        H = np.array([[last_act[c][a] / n_cls[c] * 100 for c in classes] for a in acts])
        rest_row = 100.0 - H.sum(axis=0)
        H = np.vstack([H, rest_row])
        acts = acts + ["(other)"]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12.5, 4.8), dpi=150, facecolor=SURFACE,
            gridspec_kw={"width_ratios": [1.15, 1]})
        fig.suptitle(f"{gname} group — what distinguishes the classes (train, n={sum(n_cls.values()):,})",
                     fontsize=12, color=INK, x=0.02, ha="left")

        cmap = LinearSegmentedColormap.from_list("seq_blue", BLUES)
        mesh = ax1.pcolormesh(H, cmap=cmap, vmin=0, vmax=H.max(), edgecolors=SURFACE, linewidth=2)
        ax1.set_xticks(np.arange(len(classes)) + 0.5, classes, fontsize=8.5, color=INK,
                       rotation=20, ha="right")
        ax1.set_yticks(np.arange(len(acts)) + 0.5, acts, fontsize=8.5, color=INK)
        ax1.invert_yaxis()
        ax1.set_title("last history action (% within class)", fontsize=10, color=INK_MUTED, loc="left")
        for i in range(len(acts)):
            for j in range(len(classes)):
                v = H[i, j]
                if v >= 1:
                    ax1.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                             color="#ffffff" if v > 0.6 * H.max() else INK)
        for sp in ax1.spines.values():
            sp.set_visible(False)
        ax1.tick_params(length=0)

        feats = ["turn = first", "git dirty", "files open", "ci = failed"]
        x = np.arange(len(feats))
        wbar = 0.8 / len(classes)
        for k, c in enumerate(classes):
            vals = [meta[c][f] / n_cls[c] * 100 for f in feats]
            bars = ax2.bar(x + k * wbar, vals, width=wbar - 0.03, color=CAT[k], label=c)
            for b, v in zip(bars, vals):
                ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                         fontsize=7, color=INK_MUTED)
        ax2.set_xticks(x + 0.4 - wbar / 2, feats, fontsize=8.5, color=INK)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("% of class", fontsize=9, color=INK_MUTED)
        ax2.set_title("session-state rates", fontsize=10, color=INK_MUTED, loc="left")
        ax2.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
        ax2.grid(axis="y", color="#eeedeb", linewidth=0.8)
        ax2.set_axisbelow(True)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.tick_params(length=0)

        out = f"figures/discriminators_{gname}.png"
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
        print("saved ->", out)


if __name__ == "__main__":
    main()
