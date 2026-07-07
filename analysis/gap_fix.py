"""Gap-based prediction "fixing" — does using the top1-top2 margin to alter
predictions improve macro-F1? All rules tested honestly: parameters (thresholds,
pair tables) are fit on one half of val and evaluated on the other, both ways.

Rules:
  R1 flip2   : if gap < tau, predict top-2 instead of top-1 (tau swept on fit half)
  R2 pairflip: if gap < tau, flip top1->top2 ONLY for (top1,top2) pairs where, on
               the fit half's low-gap samples, top2 is right more often than top1
  R3 arbit   : two-model — per sample take the higher-gap model's argmax (no params)
  R4 fallback: if model A's gap < tau, use model B's argmax (tau swept)

Uncalibrated logits throughout (project decision — no logit-bias calibration).
"""
import numpy as np
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, split_indices

TAUS = np.round(np.arange(0.1, 3.01, 0.1), 2)


def macro(y, p):
    return f1_score(y, p, labels=list(range(len(ALL_CLASSES))),
                    average="macro", zero_division=0)


def top12(lg):
    order = np.argsort(lg, axis=1)
    t1, t2 = order[:, -1], order[:, -2]
    gap = lg[np.arange(len(lg)), t1] - lg[np.arange(len(lg)), t2]
    return t1, t2, gap


def r1_flip2(y, t1, t2, gap, fit, ev):
    best_tau, best = 0.0, macro(y[fit], t1[fit])
    for tau in TAUS:
        p = np.where(gap[fit] < tau, t2[fit], t1[fit])
        f = macro(y[fit], p)
        if f > best:
            best, best_tau = f, tau
    p = np.where(gap[ev] < best_tau, t2[ev], t1[ev])
    return macro(y[ev], p), best_tau


def r2_pairflip(y, t1, t2, gap, fit, ev):
    best = (macro(y[fit], t1[fit]), 0.0, set())
    for tau in TAUS:
        low = fit[gap[fit] < tau]
        flip = set()
        for a in range(14):
            for b in range(14):
                m = low[(t1[low] == a) & (t2[low] == b)]
                if len(m) >= 10:
                    if (y[m] == b).sum() > (y[m] == a).sum():
                        flip.add((a, b))
        p = t1[fit].copy()
        lowm = gap[fit] < tau
        for k, i in enumerate(fit):
            if lowm[k] and (t1[i], t2[i]) in flip:
                p[k] = t2[i]
        f = macro(y[fit], p)
        if f > best[0]:
            best = (f, tau, flip)
    _, tau, flip = best
    p = t1[ev].copy()
    for k, i in enumerate(ev):
        if gap[i] < tau and (t1[i], t2[i]) in flip:
            p[k] = t2[i]
    return macro(y[ev], p), tau, len(flip)


def r4_fallback(y, t1a, gapa, t1b, fit, ev):
    best_tau, best = 0.0, macro(y[fit], t1a[fit])
    for tau in TAUS:
        p = np.where(gapa[fit] < tau, t1b[fit], t1a[fit])
        f = macro(y[fit], p)
        if f > best:
            best, best_tau = f, tau
    p = np.where(gapa[ev] < best_tau, t1b[ev], t1a[ev])
    return macro(y[ev], p), best_tau


def main():
    samples, y_all = load_samples("./data")
    _, va = split_indices(y_all, seed=42)
    y = np.array([CLASS_TO_ID[y_all[i]] for i in va])

    L = {m: np.load(f"analysis/cache/{m}_val_logits.npz")["logits"] for m in ("hist0", "qwen3")}
    T = {m: top12(L[m]) for m in L}

    rng = np.random.RandomState(0)
    fold = rng.randint(0, 2, size=len(va))
    idx = np.arange(len(va))

    base = {m: macro(y, T[m][0]) for m in L}
    print("base (uncal argmax):", {m: f"{v:.4f}" for m, v in base.items()})

    for m in L:
        t1, t2, gap = T[m]
        r1s, r2s = [], []
        for e in (0, 1):
            fit, ev = idx[fold != e], idx[fold == e]
            f1_, tau1 = r1_flip2(y, t1, t2, gap, fit, ev)
            f2_, tau2, nflip = r2_pairflip(y, t1, t2, gap, fit, ev)
            r1s.append(f1_); r2s.append(f2_)
        print(f"\n[{m}] base={base[m]:.4f}")
        print(f"  R1 flip-to-top2 (gap<tau):      heldout={np.mean(r1s):.4f}  ({np.mean(r1s)-base[m]:+.4f})")
        print(f"  R2 pair-conditional flip:       heldout={np.mean(r2s):.4f}  ({np.mean(r2s)-base[m]:+.4f})")

    # two-model rules
    t1h, _, gph = T["hist0"]; t1q, _, gpq = T["qwen3"]
    p_arb = np.where(gph >= gpq, t1h, t1q)
    print(f"\n[two-model] R3 higher-gap arbitration (no params): {macro(y, p_arb):.4f} "
          f"(vs hist0 {base['hist0']:+.4f}, vs qwen3 {base['qwen3']:+.4f})")
    for a, b, ta, tb in (("hist0", "qwen3", T["hist0"], T["qwen3"]),
                         ("qwen3", "hist0", T["qwen3"], T["hist0"])):
        scores = []
        for e in (0, 1):
            fit, ev = idx[fold != e], idx[fold == e]
            f, tau = r4_fallback(y, ta[0], ta[2], tb[0], fit, ev)
            scores.append(f)
        print(f"[two-model] R4 {a}, fallback->{b} when gap<tau: heldout={np.mean(scores):.4f} "
              f"(vs {a} base {np.mean(scores)-base[a]:+.4f})")

    # oracle ceiling for context: pick whichever model is right (upper bound)
    oracle = np.where(t1h == y, t1h, np.where(t1q == y, t1q, t1h))
    print(f"\noracle either-right ceiling: {macro(y, oracle):.4f}")


if __name__ == "__main__":
    main()


def perclass_variants():
    """Per-class tau: fit tau_c for each PREDICTED class c on the fit half."""
    samples, y_all = load_samples("./data")
    _, va = split_indices(y_all, seed=42)
    y = np.array([CLASS_TO_ID[y_all[i]] for i in va])
    L = {m: np.load(f"analysis/cache/{m}_val_logits.npz")["logits"] for m in ("hist0", "qwen3")}
    T = {m: top12(L[m]) for m in L}
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 2, size=len(va))
    idx = np.arange(len(va))
    base = {m: macro(y, T[m][0]) for m in L}

    def fit_perclass_tau(y, t1, alt, gap, fit):
        """greedy per-class tau: for each predicted class, pick tau maximizing
        fit-half macro-F1 with other classes' taus fixed (one pass)."""
        taus = np.zeros(14)
        p = t1[fit].copy()
        best = macro(y[fit], p)
        for c in range(14):
            cur_best, cur_tau = best, 0.0
            for tau in TAUS:
                p_try = p.copy()
                m = (t1[fit] == c) & (gap[fit] < tau)
                p_try[m] = alt[fit][m]
                f = macro(y[fit], p_try)
                if f > cur_best:
                    cur_best, cur_tau = f, tau
            taus[c] = cur_tau
            m = (t1[fit] == c) & (gap[fit] < cur_tau)
            p[m] = alt[fit][m]
            best = cur_best
        return taus

    def apply_perclass(t1, alt, gap, ev, taus):
        p = t1[ev].copy()
        for c in range(14):
            m = (t1[ev] == c) & (gap[ev] < taus[c])
            p[m] = alt[ev][m]
        return p

    print("=== per-class tau variants (held-out 2-fold) ===")
    for m in L:
        t1, t2, gap = T[m]
        scores = []
        for e in (0, 1):
            fit, ev = idx[fold != e], idx[fold == e]
            taus = fit_perclass_tau(y, t1, t2, gap, fit)
            scores.append(macro(y[ev], apply_perclass(t1, t2, gap, ev, taus)))
        print(f"[{m}] R1' per-class flip-to-top2: heldout={np.mean(scores):.4f} ({np.mean(scores)-base[m]:+.4f})")

    for a, b in (("qwen3", "hist0"), ("hist0", "qwen3")):
        t1a, _, gapa = T[a]; t1b, _, _ = T[b]
        scores = []
        for e in (0, 1):
            fit, ev = idx[fold != e], idx[fold == e]
            taus = fit_perclass_tau(y, t1a, t1b, gapa, fit)
            scores.append(macro(y[ev], apply_perclass(t1a, t1b, gapa, ev, taus)))
        print(f"[{a}] R4' per-class fallback->{b}: heldout={np.mean(scores):.4f} ({np.mean(scores)-base[a]:+.4f})")


if __name__ == "__main__" and __import__("sys").argv[-1] == "perclass":
    perclass_variants()
