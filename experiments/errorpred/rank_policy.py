"""E27 phase 1b — per-(decile × predicted-class) rank-selection policy.

User ask 2026-07-09: based on the decile statistics, compute WHICH rank should be
selected as the prediction, for each class × decile. Conditioning is on observables
only (MSP decile + rank-1 predicted class — the true class is unknown at test time).

Policy: for cell (d, c) = rows in MSP-decile d whose top-1 class is c, pick the rank
k∈{1..4} with the highest empirical P(true = rank-k). Guards: cells with n < min_n
keep rank 1. Honest evaluation = 2-fold cross-fit (fit cells on one half, apply to the
other; E6 lesson — in-sample rule mining flatters).

CPU-only, cached logits. Raw fp32, NO calibration.

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/rank_policy.py
"""
import argparse

import numpy as np

from src.data import ALL_CLASSES

CACHE = "analysis/cache/clpvi_tiers_granite_ls.npz"
SHORT = ["read", "grep", "lsdir", "glob", "edit", "write", "patch",
         "bash", "tests", "lint", "ask", "plan", "web", "resp"]


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def fit_policy(dec_id, pred, rank_true, rows, C, min_n, K=4):
    """best rank per (decile, pred-class) cell, fitted on `rows` only. 0-based ranks."""
    best = np.zeros((10, C), int)                       # default rank 1 (index 0)
    for d in range(10):
        for c in range(C):
            m = rows[(dec_id[rows] == d) & (pred[rows] == c)]
            if len(m) < min_n:
                continue
            accs = [(rank_true[m] == k).mean() for k in range(K)]
            best[d, c] = int(np.argmax(accs))
    return best


def apply_policy(best, dec_id, pred, order, rows):
    k = best[dec_id[rows], pred[rows]]
    return order[rows, k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_n", type=int, default=20)
    args = ap.parse_args()

    D = np.load(CACHE)
    z, y = D["logits"], D["labels"]
    C = len(ALL_CLASSES)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    order = np.argsort(-z, 1)
    pred = order[:, 0]
    rank_true = np.argmax(order == y[:, None], 1)
    n = len(y)
    qs = np.quantile(msp, np.linspace(0, 1, 11))
    dec_id = np.clip(np.searchsorted(qs, msp, side="right") - 1, 0, 9)

    # ---- in-sample: fit + apply on all of val (upper bound / structure read) ----
    allrows = np.arange(n)
    best = fit_policy(dec_id, pred, rank_true, allrows, C, args.min_n)
    sel = apply_policy(best, dec_id, pred, order, allrows)
    base_acc, pol_acc = (pred == y).mean(), (sel == y).mean()
    print("## E27-1b · which rank wins, per (MSP decile × PREDICTED class)  [in-sample]\n")
    print("best rank per cell (rows = decile 1..10 asc confidence; '.' = rank1 or n<20):\n")
    print("| dec | " + " | ".join(SHORT) + " |")
    print("|---|" + "---|" * C)
    for d in range(10):
        cells = []
        for c in range(C):
            m = (dec_id == d) & (pred == c)
            cells.append("." if best[d, c] == 0 or m.sum() < args.min_n
                         else f"**{best[d, c] + 1}**")
        print(f"| {d+1} | " + " | ".join(cells) + " |")

    print("\ncells where a rank ≠ 1 wins in-sample (n ≥ 20):\n")
    print("| decile | pred class | n | acc rank1 | acc best rank | Δ |")
    print("|---|---|---|---|---|---|")
    tot_gain = 0.0
    for d in range(10):
        for c in range(C):
            if best[d, c] == 0:
                continue
            m = np.where((dec_id == d) & (pred == c))[0]
            a1 = (rank_true[m] == 0).mean()
            ab = (rank_true[m] == best[d, c]).mean()
            tot_gain += (ab - a1) * len(m)
            print(f"| {d+1} | {SHORT[c]} | {len(m)} | {a1:.3f} | {ab:.3f} | +{ab - a1:.3f} |")
    print(f"\nin-sample: base acc {base_acc:.4f} · policy acc {pol_acc:.4f} "
          f"(Δ {pol_acc - base_acc:+.4f}, {tot_gain:.0f} rows) · "
          f"base mF1 {mf1(y, pred):.4f} · policy mF1 {mf1(y, sel):.4f}")

    # ---- honest read: 2-fold cross-fit (fit on half, apply to the other) ----
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    half = [perm[: n // 2], perm[n // 2:]]
    sel_cf = pred.copy()
    for f in (0, 1):
        b = fit_policy(dec_id, pred, rank_true, half[f], C, args.min_n)
        sel_cf[half[1 - f]] = apply_policy(b, dec_id, pred, order, half[1 - f])
        nz = (b != 0).sum()
        print(f"fold {f}: {nz} non-rank1 cells fitted")
    print(f"cross-fitted: base acc {base_acc:.4f} · policy acc {(sel_cf == y).mean():.4f} "
          f"(Δ {(sel_cf == y).mean() - base_acc:+.4f}) · "
          f"base mF1 {mf1(y, pred):.4f} · policy mF1 {mf1(y, sel_cf):.4f} "
          f"(Δ {mf1(y, sel_cf) - mf1(y, pred):+.4f})")


if __name__ == "__main__":
    main()
