"""E27 phase 1c-verify — are the (rank1, rank2, decile) flip cells real?

Two checks (user GO 2026-07-10), both CPU on cached logits:
  1. Cross-fitted swap policy: 2-fold (5 seeds) — fit "swap to rank2" cells
     (n>=min_n and P(true=r2) > P(true=r1)) on one half, apply to the other.
     Reported per model: Δacc / Δmacro-F1 vs baseline argmax.
  2. Cross-model replication of the named flip cells (deciles = each model's OWN
     MSP quantiles): if prior-dominance is real, every same-prior model flips too.

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/pair_policy_check.py
"""
import numpy as np

from src.data import ALL_CLASSES, CLASS_TO_ID

SHORT = ["read", "grep", "lsdir", "glob", "edit", "write", "patch",
         "bash", "tests", "lint", "ask", "plan", "web", "resp"]
CELLS = [("read", "lsdir", 0), ("grep", "lsdir", 0), ("bash", "tests", 1),
         ("plan", "ask", 1), ("grep", "read", 0), ("grep", "read", 1)]
MIN_N = 20


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def prep(z):
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    order = np.argsort(-z, 1)
    qs = np.quantile(msp, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(qs, msp, side="right") - 1, 0, 9)
    return order[:, 0], order[:, 1], dec


def crossfit(y, r1, r2, dec, seeds=(42, 1, 2, 3, 4)):
    n = len(y)
    base_acc, base_f1 = None, None
    d_acc, d_f1, ncells = [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(n)
        halves = [perm[: n // 2], perm[n // 2:]]
        pred = r1.copy()
        tot_cells = 0
        for f in (0, 1):
            fit, app = halves[f], halves[1 - f]
            key_fit = (r1[fit] * 14 + r2[fit]) * 10 + dec[fit]
            key_app = (r1[app] * 14 + r2[app]) * 10 + dec[app]
            for key in np.unique(key_fit):
                m = fit[key_fit == key]
                if len(m) < MIN_N:
                    continue
                a, b = (key // 10) // 14, (key // 10) % 14
                if (y[m] == b).mean() > (y[m] == a).mean():
                    pred[app[key_app == key]] = b
                    tot_cells += 1
        if base_acc is None:
            base_acc, base_f1 = (r1 == y).mean(), mf1(y, r1)
        d_acc.append((pred == y).mean() - base_acc)
        d_f1.append(mf1(y, pred) - base_f1)
        ncells.append(tot_cells)
    return base_acc, base_f1, np.array(d_acc), np.array(d_f1), ncells


def main():
    D = np.load("analysis/cache/clpvi_tiers_granite_ls.npz")
    y = D["labels"]
    models = {"granite_ls(e9)": D["logits"]}
    models.update(dict(np.load("analysis/cache/coreset_gate_logits.npz")))
    models["qwen3"] = np.load("analysis/cache/qwen3_val_logits.npz")["logits"]

    print("## Check 2 · flip-cell replication across models (deciles = each model's own MSP)\n")
    print("| cell (r1→r2, dec) | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for a_s, b_s, d in CELLS:
        a, b = CLASS_TO_ID_S[a_s], CLASS_TO_ID_S[b_s]
        row = []
        for name, z in models.items():
            r1, r2, dec = prep(z)
            m = (r1 == a) & (r2 == b) & (dec == d)
            if m.sum() < 10:
                row.append(f"n={int(m.sum())}")
                continue
            p1, p2 = (y[m] == a).mean(), (y[m] == b).mean()
            mark = "✅" if p2 > p1 else "❌"
            row.append(f"{mark} {p1:.2f}/{p2:.2f} (n={int(m.sum())})")
        print(f"| {a_s}→{b_s} d{d+1} | " + " | ".join(row) + " |")
    print("\n(cell = P(true=r1)/P(true=r2); ✅ = rank2 wins)\n")

    print("## Check 1 · cross-fitted swap policy (2-fold × 5 seeds), per model\n")
    print("| model | base acc | base mF1 | Δacc mean [min,max] | ΔmF1 mean [min,max] | swap-cells/run |")
    print("|---|---|---|---|---|---|")
    for name, z in models.items():
        r1, r2, dec = prep(z)
        ba, bf, da, df, nc = crossfit(y, r1, r2, dec)
        print(f"| {name} | {ba:.4f} | {bf:.4f} | {da.mean():+.4f} [{da.min():+.4f},{da.max():+.4f}] "
              f"| {df.mean():+.4f} [{df.min():+.4f},{df.max():+.4f}] | {min(nc)}–{max(nc)} |")


CLASS_TO_ID_S = {s: i for i, s in enumerate(SHORT)}

if __name__ == "__main__":
    main()
