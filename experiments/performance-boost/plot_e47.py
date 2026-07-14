"""E47 wave-1 figure: per-epoch 14k-val curves + best-epoch bars vs anchor."""
import matplotlib.pyplot as plt
import numpy as np

BG, BLUE, RED, GRID = "#fcfcfb", "#3987e5", "#d4553f", "#dddddd"
curves = {
    "anchor (AWP)":   [.6561, .7532, .7782, .7750, .7747, .7736],
    "A1 neg-LS":      [.6055, .7041, .7433, .7615, .7783, .7759],
    "A2 grouped-LS":  [.5046, .6824, .7358, .7589, .7767, .7766],
    "A3 DropHead":    [.6554, .7468, .7799, .7795, .7826, .7773],
    "A4 Child-Tune":  [.6608, .7565, .7783, .7741, .7780, .7754],
    "A5 MSD head":    [.6406, .7650, .7792, .7744, .7753, .7730],
}
anchor_best = 0.7782
ep = np.arange(1, 7)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.3, 1]})
for ax in (axL, axR):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
colors = {"A3 DropHead": RED, "anchor (AWP)": "#222222"}
for name, ys in curves.items():
    c = colors.get(name, BLUE)
    lw = 2.6 if name in colors else 1.4
    a = 1.0 if name in colors else 0.55
    axL.plot(ep, ys, "-o", color=c, lw=lw, alpha=a, ms=4, label=name)
axL.set_xlabel("epoch"); axL.set_ylabel("14k-val macro-F1")
axL.set_title("E47 per-epoch 14k-val curves (56k/14k)", fontsize=11)
axL.grid(True, color=GRID, lw=0.6, alpha=0.6); axL.set_ylim(.60, .79)
axL.legend(fontsize=8, framealpha=0.9)

names = ["A2 grouped-LS", "A1 neg-LS", "A4 Child-Tune", "anchor (AWP)", "A5 MSD head", "A3 DropHead"]
best = {"anchor (AWP)": .7782, "A1 neg-LS": .7783, "A2 grouped-LS": .7767,
        "A3 DropHead": .7826, "A4 Child-Tune": .7783, "A5 MSD head": .7792}
vals = [best[n] for n in names]
cols = [RED if n == "A3 DropHead" else ("#222222" if n == "anchor (AWP)" else BLUE) for n in names]
y = np.arange(len(names))
axR.barh(y, vals, color=cols, alpha=0.85)
axR.axvline(anchor_best, color="#222222", ls="--", lw=1, alpha=0.7)
axR.set_yticks(y); axR.set_yticklabels(names, fontsize=9)
axR.set_xlim(.775, .784); axR.set_xlabel("best-epoch 14k-val macro-F1")
axR.set_title("best epoch vs anchor (dashed)", fontsize=11)
for yi, v in zip(y, vals):
    d = v - anchor_best
    axR.text(v + 0.0002, yi, f"{v:.4f} ({d:+.4f})", va="center", fontsize=8)
axR.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig("experiments/performance-boost/figures/e47_levers.png", dpi=130, facecolor=BG)
print("saved figures/e47_levers.png")
