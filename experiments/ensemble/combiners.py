"""E26 — compare ENSEMBLE COMBINER types on the cached 3.5k-slice logits (CPU, seconds).

Given the member pool (cached by screen_ensemble.py), re-derives the greedy member set under
uniform soft voting, then scores EVERY cheap combiner on (a) the best pair, (b) the greedy set:

  soft-uniform   mean of member softmax probs (the project default — 0 fitted params)
  logit-mean     mean of raw logits (scale-sensitive across LS/CE members)
  geo-mean       product of experts: mean of log-probs
  hard-vote      majority over member argmax (tie -> soft-uniform winner)
  rank-mean      per member, rank classes by prob; average ranks
  weighted       prob weights fit by coordinate ascent — ⚠️ fitted on the slice; reported as
                 2-fold honest CV (fit half A, eval half B, swap, mean) AND full-slice (overfit
                 upper bound). Calibration-adjacent: NOT for shipping without user sign-off.
  stacking-LR    logistic regression on concat member probs — same 2-fold honest protocol.

  PYTHONPATH=. python experiments/ensemble/combiners.py [--members t1,t2,...]
"""
import argparse
import glob

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, split_indices

CACHE = "analysis/cache/e26_screen_logits.npz"


def sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="", help="comma tags; default = greedy set (auto)")
    ap.add_argument("--greedy_len", type=int, default=8)
    args = ap.parse_args()

    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    samples, labels = load_samples("./data")
    y = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y[va], random_state=42)
    yt = y[va_eval]

    def mf1(pred):
        return f1_score(yt, pred, labels=np.arange(len(ALL_CLASSES)), average="macro",
                        zero_division=0)

    cache = {}
    for p in [CACHE] + sorted(glob.glob(CACHE.replace(".npz", "_s*.npz"))):
        try:
            d = np.load(p)
            cache.update({k: d[k] for k in d.files})
        except FileNotFoundError:
            pass
    L = {t: v.astype(np.float64) for t, v in cache.items()}
    P = {t: sm(v) for t, v in L.items()}
    tags = sorted(P)
    logger.info(f"{len(tags)} cached members; slice n={len(yt)}")

    def uni(ms):
        return mf1(np.mean([P[t] for t in ms], 0).argmax(1))

    singles = sorted(((mf1(P[t].argmax(1)), t) for t in tags), reverse=True)
    best_pair = max(((uni([a, b]), a, b) for i, a in enumerate(tags) for b in tags[i + 1:]))
    if args.members:
        greedy = args.members.split(",")
    else:
        greedy, pool, cur = [], set(tags), 0.0
        for _ in range(args.greedy_len):
            f, t = max((uni(greedy + [t]), t) for t in pool)
            if greedy and f <= cur:
                break
            greedy.append(t); pool.discard(t); cur = f
    print(f"\nbest single: {singles[0][1]} {singles[0][0]:.4f}")
    print(f"best pair (soft-uniform): {best_pair[1]} + {best_pair[2]} = {best_pair[0]:.4f}")
    print(f"greedy set (soft-uniform): {greedy} = {uni(greedy):.4f}")

    half_a, half_b = train_test_split(np.arange(len(yt)), test_size=0.5,
                                      stratify=yt, random_state=0)

    def fit_weights(ms, rows):
        w = np.ones(len(ms)) / len(ms)
        stack = np.stack([P[t][rows] for t in ms])          # (M, n, C)
        yr = yt[rows]
        def f1w(w_):
            pred = np.tensordot(w_ / w_.sum(), stack, 1).argmax(1)
            return f1_score(yr, pred, labels=np.arange(len(ALL_CLASSES)),
                            average="macro", zero_division=0)
        best = f1w(w)
        for _ in range(4):
            improved = False
            for m in range(len(ms)):
                for v in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                    w2 = w.copy(); w2[m] = v
                    if w2.sum() == 0:
                        continue
                    f = f1w(w2)
                    if f > best + 1e-6:
                        best, w, improved = f, w2, True
            if not improved:
                break
        return w / w.sum()

    def eval_all(ms, name):
        stackP = np.stack([P[t] for t in ms])
        stackL = np.stack([L[t] for t in ms])
        rows = []
        rows.append(("soft-uniform", mf1(stackP.mean(0).argmax(1))))
        rows.append(("logit-mean", mf1(stackL.mean(0).argmax(1))))
        rows.append(("geo-mean", mf1(np.log(np.clip(stackP, 1e-12, 1)).mean(0).argmax(1))))
        votes = stackP.argmax(2)                           # (M, n)
        maj = np.zeros_like(yt)
        soft = stackP.mean(0)
        for i in range(len(yt)):
            c = np.bincount(votes[:, i], minlength=len(ALL_CLASSES))
            top = np.flatnonzero(c == c.max())
            maj[i] = top[0] if len(top) == 1 else max(top, key=lambda k: soft[i, k])
        rows.append(("hard-vote", mf1(maj)))
        ranks = stackP.argsort(2).argsort(2).mean(0)       # mean rank per class
        rows.append(("rank-mean", mf1(ranks.argmax(1))))
        # fitted combiners — 2-fold honest + full-slice upper bound
        wa = fit_weights(ms, half_a); wb = fit_weights(ms, half_b)
        pa = np.tensordot(wb, stackP[:, half_a], 1).argmax(1)
        pb = np.tensordot(wa, stackP[:, half_b], 1).argmax(1)
        hon = (f1_score(yt[half_a], pa, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)
               + f1_score(yt[half_b], pb, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)) / 2
        wfull = fit_weights(ms, np.arange(len(yt)))
        rows.append((f"weighted ⚠️ (2-fold honest; w={np.round(wfull,2).tolist()})", hon))
        rows.append(("weighted ⚠️ (full-slice UPPER BOUND)",
                     mf1(np.tensordot(wfull, stackP, 1).argmax(1))))
        from sklearn.linear_model import LogisticRegression
        X = np.concatenate([P[t] for t in ms], 1)
        f_st = []
        for tr_r, ev_r in ((half_a, half_b), (half_b, half_a)):
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr_r], yt[tr_r])
            f_st.append(f1_score(yt[ev_r], lr.predict(X[ev_r]),
                                 labels=np.arange(len(ALL_CLASSES)), average="macro",
                                 zero_division=0))
        rows.append(("stacking-LR ⚠️ (2-fold honest)", float(np.mean(f_st))))
        print(f"\n## Combiners on {name}: {ms}\n")
        print("| combiner | macro-F1 |\n|---|---|")
        for n, f in rows:
            print(f"| {n} | {f:.4f} |")

    eval_all([best_pair[1], best_pair[2]], "best pair")
    if len(greedy) > 2:
        eval_all(greedy, "greedy set")


if __name__ == "__main__":
    main()
