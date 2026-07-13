"""Robust ensemble selection over experiments/ensemble/logits/ — methods A and C.

Combiner is ALWAYS uniform softmax mean (E29: uniform mean beats all learned combiners).
These are SELECTION procedures, not combiners.

A — bootstrap-bagged greedy (winner's-curse-robust):
    * B stratified bootstrap resamples of the 3.5k rows, drawn ONCE and shared across all
      combos (paired / common random numbers) — see build_bags().
    * run greedy forward selection (uniform, no replacement) INSIDE each bag.
    * member weight = fraction of bags that selected it; the robust set = members above --freq.
    * every combo also gets a bootstrap F1 distribution -> rank by lower bound, not the point max.

C — diversity-regularized greedy (single full-slice pass):
    * greedy that maximizes  F1(ensemble)  +  lambda * mean_disagreement(candidate, current set)
    * rewards members that ERR on different rows (the mean only pays off on error diversity).

Prints both selected sets + their full-slice F1 and bootstrap CI so they can be compared head-to-head.

  PYTHONPATH=. python experiments/ensemble/select_ensemble.py --B 1000 --lam 0.05
"""
import argparse
import glob
import os

import numpy as np

D = "experiments/ensemble/logits"
K = 14


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_f1_fast(t, p):
    tp = np.bincount(t[t == p], minlength=K)
    fp = np.bincount(p, minlength=K) - tp
    fn = np.bincount(t, minlength=K) - tp
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / denom, 0.0).mean()


def build_bags(y, B, seed=42):
    """B stratified bootstrap index sets, drawn once, shared across all combos.
    Stratified: resample WITHIN each class so per-class counts are preserved (no class
    vanishes from a resample -> clean macro-F1 variance)."""
    rng = np.random.default_rng(seed)
    class_pos = [np.where(y == c)[0] for c in range(K)]
    class_pos = [p for p in class_pos if len(p)]
    bags = np.empty((B, len(y)), dtype=np.int32)
    for b in range(B):
        bags[b] = np.concatenate([rng.choice(p, size=len(p), replace=True) for p in class_pos])
    return bags


def load_pool(include, exclude):
    meta = np.load(f"{D}/_meta.npz")
    y = meta["labels"].astype(int)
    inc = [s for s in include.split(",") if s]
    exc = [s for s in exclude.split(",") if s]
    P = {}
    for f in sorted(glob.glob(f"{D}/*.npz")):
        t = os.path.basename(f)[:-4]
        if t == "_meta":
            continue
        if inc and not any(s in t for s in inc):
            continue
        if exc and any(s in t for s in exc):
            continue
        P[t] = softmax(np.load(f)["logits"])
    return y, P


def ens_pred(P, members):
    return (sum(P[t] for t in members) / len(members)).argmax(1)


def method_A(y, P, B, greedy_len, freq_thresh, seed):
    bags = build_bags(y, B, seed)
    tags = list(P)
    preds = {t: P[t].argmax(1) for t in tags}          # single-model preds (unused directly)
    counts = {t: 0 for t in tags}
    for b in range(B):
        bi = bags[b]
        yb = y[bi]
        members, cur, pool = [], -1.0, set(tags)
        # cache running prob-sum on the bag rows for speed
        run = None
        for _ in range(greedy_len):
            best_f, best_t, best_run = cur, None, None
            for t in pool:
                pb = P[t][bi]
                cand = pb if run is None else run + pb
                f = macro_f1_fast(yb, (cand / (len(members) + 1)).argmax(1))
                if f > best_f:
                    best_f, best_t, best_run = f, t, cand
            if best_t is None:
                break
            members.append(best_t); pool.discard(best_t); run = best_run; cur = best_f
        for t in members:
            counts[t] += 1
    freq = {t: counts[t] / B for t in tags}
    robust = [t for t in sorted(freq, key=freq.get, reverse=True) if freq[t] >= freq_thresh]
    return robust, freq, bags


def method_C(y, P, greedy_len, lam):
    tags = list(P)
    preds = {t: P[t].argmax(1) for t in tags}
    members, cur, pool = [], -1.0, set(tags)
    trace = []
    for _ in range(greedy_len):
        best_score, best_t, best_f1 = -1e9, None, None
        cur_pred = None if not members else ens_pred(P, members)
        for t in pool:
            f1 = macro_f1_fast(y, ens_pred(P, members + [t]))
            if members:
                # mean pairwise disagreement of candidate vs current members' argmax
                div = np.mean([(preds[t] != preds[m]).mean() for m in members])
            else:
                div = 0.0
            score = f1 + lam * div
            if score > best_score:
                best_score, best_t, best_f1 = score, t, f1
        if best_t is None or (members and best_f1 < cur - 0.01):
            break
        members.append(best_t); pool.discard(best_t); cur = best_f1
        trace.append((best_t, best_f1))
    return members, trace


def ci(y, P, members, bags):
    pred = ens_pred(P, members)
    dist = np.array([macro_f1_fast(y[bi], pred[bi]) for bi in bags])
    return macro_f1_fast(y, pred), np.percentile(dist, [10, 50, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--greedy_len", type=int, default=8)
    ap.add_argument("--freq", type=float, default=0.5, help="A: min bag-selection frequency")
    ap.add_argument("--lam", type=float, default=0.05, help="C: diversity weight")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include", default="")
    ap.add_argument("--exclude", default="")
    args = ap.parse_args()

    y, P = load_pool(args.include, args.exclude)
    print(f"pool: {len(P)} models on n={len(y)}   B={args.B}\n")

    # baseline: plain greedy (current method) for reference
    base, btrace = method_C(y, P, args.greedy_len, 0.0)   # lam=0 == plain score greedy

    robust, freq, bags = method_A(y, P, args.B, args.greedy_len, args.freq, args.seed)
    cmembers, ctrace = method_C(y, P, args.greedy_len, args.lam)

    print("== member bag-frequency (method A, top 15) ==")
    for t in sorted(freq, key=freq.get, reverse=True)[:15]:
        print(f"  {freq[t]:.2f}  {t}")

    print(f"\n== A robust set (freq>={args.freq}, {len(robust)}): {robust}")
    print(f"== C diverse set (lam={args.lam}, {len(cmembers)}): {cmembers}")
    print(f"== baseline plain-greedy set: {[t for t,_ in btrace]}")

    print("\n== head-to-head (full-slice F1 + bootstrap CI[10/50/90]) ==")
    for name, ms in [("baseline greedy", [t for t, _ in btrace]),
                     ("A bagged-robust", robust),
                     ("C diversity-reg", cmembers)]:
        if not ms:
            print(f"  {name:18s}: (empty)"); continue
        f, (lo, md, hi) = ci(y, P, ms, bags)
        print(f"  {name:18s}: F1={f:.4f}  CI[{lo:.4f}, {md:.4f}, {hi:.4f}]  ({len(ms)} members)")


if __name__ == "__main__":
    main()
