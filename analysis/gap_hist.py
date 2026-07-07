"""Gap vs history length: figure + history-conditioned tau fallback test.

Figure: median gap (correct vs wrong) across history-length buckets, both models.
Test:   R4'' — qwen3 -> hist0 fallback with tau conditioned on (pred class x
        zero-vs-has-history), vs per-class-only (0.7732) and global (0.7702).
        Same honest 2-fold protocol; uncalibrated logits.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, split_indices

BLUE, RED = "#3987e5", "#d4553f"
INK, SUB, GRID, BG = "#0b0b0b", "#52514e", "#e4e2dd", "#fcfcfb"
BUCKETS = [(0, 0, "0"), (1, 3, "1-3"), (4, 6, "4-6"), (7, 9, "7-9"), (10, 99, "10+")]
TAUS = np.round(np.arange(0.1, 3.01, 0.1), 2)


def macro(y, p):
    return f1_score(y, p, labels=list(range(14)), average="macro", zero_division=0)


def top12(lg):
    o = np.argsort(lg, axis=1)
    t1, t2 = o[:, -1], o[:, -2]
    return t1, t2, lg[np.arange(len(lg)), t1] - lg[np.arange(len(lg)), t2]


def fig(models, gaps, corrs, hlen):
    x = np.arange(len(BUCKETS))
    fig_, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), facecolor=BG, sharey=True)
    for ax, m in zip(axes, models):
        gap, corr = gaps[m], corrs[m]
        med_c, med_w = [], []
        for lo, hi, _ in BUCKETS:
            msk = (hlen >= lo) & (hlen <= hi)
            med_c.append(np.median(gap[msk & corr]))
            med_w.append(np.median(gap[msk & ~corr]))
        ax.set_facecolor(BG)
        ax.plot(x, med_c, "o-", color=BLUE, lw=2, ms=6, label="correct", zorder=3)
        ax.plot(x, med_w, "o-", color=RED, lw=2, ms=6, label="incorrect", zorder=3)
        for xi, (c, w) in enumerate(zip(med_c, med_w)):
            ax.text(xi, c + .12, f"{c:.1f}", ha="center", fontsize=8, color=INK)
            ax.text(xi, w - .3, f"{w:.1f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [b[2] for b in BUCKETS], fontsize=9)
        ax.set_xlabel("history length", fontsize=9, color=SUB)
        ax.set_title(models[m], fontsize=10.5, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(labelsize=8.5, colors=SUB)
    axes[0].set_ylabel("median top1−top2 gap (uncal)", fontsize=9, color=SUB)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig_.suptitle("Decision margin vs history length — separation collapses at history 0",
                  fontsize=12, color=INK, x=0.02, ha="left")
    fig_.tight_layout(rect=[0, 0, 1, 0.91])
    out = "figures/gap_by_histlen.png"
    fig_.savefig(out, dpi=170, facecolor=BG)
    print("saved ->", out)


def fit_tau(y, t1, alt, gap, fit, keys, keyspace):
    """greedy tau per key (key = class or (class,zh)); one pass."""
    taus = {k: 0.0 for k in keyspace}
    p = t1[fit].copy()
    best = macro(y[fit], p)
    for k in keyspace:
        cur_best, cur_tau = best, 0.0
        for tau in TAUS:
            p_try = p.copy()
            m = (keys[fit] == k) & (gap[fit] < tau) if keys.ndim == 1 else None
            m = (keys[fit] == k) & (gap[fit] < tau)
            p_try[m] = alt[fit][m]
            f = macro(y[fit], p_try)
            if f > cur_best:
                cur_best, cur_tau = f, tau
        taus[k] = cur_tau
        m = (keys[fit] == k) & (gap[fit] < cur_tau)
        p[m] = alt[fit][m]
        best = cur_best
    return taus


def apply_tau(t1, alt, gap, ev, keys, taus):
    p = t1[ev].copy()
    for k, tau in taus.items():
        m = (keys[ev] == k) & (gap[ev] < tau)
        p[m] = alt[ev][m]
    return p


def main():
    samples, y_all = load_samples("./data")
    _, va = split_indices(y_all, seed=42)
    y = np.array([CLASS_TO_ID[y_all[i]] for i in va])
    hlen = np.array([len(samples[i]["history"]) for i in va])
    zh = (hlen == 0).astype(int)

    models = {"hist0": "hist0 (bge-m3 full-FT)", "qwen3": "Qwen3-0.6B full-FT"}
    L = {m: np.load(f"analysis/cache/{m}_val_logits.npz")["logits"] for m in models}
    T = {m: top12(L[m]) for m in models}
    fig(models, {m: T[m][2] for m in models}, {m: T[m][0] == y for m in models}, hlen)

    # ---- R4'' : qwen3 -> hist0 fallback, tau per (class x zero-history) ----
    t1q, _, gapq = T["qwen3"]
    t1h = T["hist0"][0]
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 2, size=len(va))
    idx = np.arange(len(va))
    base = macro(y, t1q)

    variants = {
        "per-class (28->14 keys)": (t1q, list(range(14))),
        "per-(class x zerohist)": (t1q * 2 + zh, list(range(28))),
        "per-zerohist only (2 keys)": (zh, [0, 1]),
    }
    print(f"\nqwen3 base={base:.4f}; fallback->hist0, held-out 2-fold:")
    for name, (keys, keyspace) in variants.items():
        scores = []
        for e in (0, 1):
            fit, ev = idx[fold != e], idx[fold == e]
            taus = fit_tau(y, t1q, t1h, gapq, fit, keys, keyspace)
            scores.append(macro(y[ev], apply_tau(t1q, t1h, gapq, ev, keys, taus)))
        print(f"  {name:28} {np.mean(scores):.4f} ({np.mean(scores)-base:+.4f})")


if __name__ == "__main__":
    main()
