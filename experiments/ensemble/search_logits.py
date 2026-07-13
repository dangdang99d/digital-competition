"""Search ensemble combinations over experiments/ensemble/logits/ (one .npz per model).

Loads _meta.npz (shared 3.5k held-out slice) + every <tag>.npz, then scores by UNIFORM
softmax mean (project invariant — no weights, no calibration):
  singles, top pairs/triples, greedy forward selection (Caruana, uniform, no replacement).

  PYTHONPATH=. python experiments/ensemble/search_logits.py --greedy_len 6 --topk 20
  # restrict the pool (substring match, comma-sep) e.g. only richargs granite champions:
  PYTHONPATH=. python experiments/ensemble/search_logits.py --include e34,e8a,e25c,e32
"""
import argparse
import glob
import itertools
import os

import numpy as np
from sklearn.metrics import f1_score

D = "experiments/ensemble/logits"


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy_len", type=int, default=6)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--triples", action="store_true", help="also brute-force top triples")
    ap.add_argument("--include", default="", help="comma-sep substrings; keep only matching tags")
    ap.add_argument("--exclude", default="", help="comma-sep substrings; drop matching tags")
    args = ap.parse_args()

    meta = np.load(f"{D}/_meta.npz")
    y = meta["labels"]
    def mf1(p): return f1_score(y, p, labels=np.arange(14), average="macro", zero_division=0)

    inc = [s for s in args.include.split(",") if s]
    exc = [s for s in args.exclude.split(",") if s]
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
    tags = list(P)
    print(f"pool: {len(tags)} models on n={len(y)}")

    singles = sorted(((mf1(P[t].argmax(1)), t) for t in tags), reverse=True)
    best = singles[0][0]
    print(f"\nbest single: {singles[0][1]} {best:.4f}")

    def ens(ms): return mf1((sum(P[t] for t in ms) / len(ms)).argmax(1))

    pairs = sorted((ens([a, b]), a, b) for a, b in itertools.combinations(tags, 2))
    print(f"\ntop {args.topk} pairs:")
    for f, a, b in pairs[::-1][:args.topk]:
        print(f"  {f:.4f} ({f-best:+.4f})  {a} + {b}")

    if args.triples:
        tri = sorted((ens([a, b, c]), a, b, c)
                     for a, b, c in itertools.combinations([t for _, t in singles[:24]], 3))
        print(f"\ntop {args.topk} triples (from top-24 singles):")
        for f, a, b, c in tri[::-1][:args.topk]:
            print(f"  {f:.4f} ({f-best:+.4f})  {a} + {b} + {c}")

    print("\ngreedy (Caruana, uniform, no replacement):")
    members, cur, pool = [], 0.0, set(tags)
    for step in range(args.greedy_len):
        f_best, t_best = max((ens(members + [t]), t) for t in pool)
        if members and f_best <= cur:
            print(f"  stop @ {cur:.4f}")
            break
        members.append(t_best); pool.discard(t_best)
        print(f"  +{t_best:32s} -> {f_best:.4f} ({f_best-cur:+.4f})")
        cur = f_best
    print(f"\ngreedy set ({len(members)}): {members}  = {cur:.4f}")


if __name__ == "__main__":
    main()
