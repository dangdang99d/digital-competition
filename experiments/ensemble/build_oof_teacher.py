"""E30 phase 1 — OOF teacher from the merged member caches.

teacher = uniform mean of the members' OOF probs per row (the honest analogue of the
E26 trio teacher — 1B failed because in-sample harvest gave near-one-hot targets).
Saved as npz{logits: log(mean_prob)} aligned to load_samples order — log-probs are
valid teacher "logits" for finetune.py's KD (softmax(log p / T) = tempered p).

  python experiments/ensemble/build_oof_teacher.py \
      --members is3,aum06,e25c --out analysis/cache/e30_teacher_oof.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="is3,aum06,e25c")
    ap.add_argument("--out", default="analysis/cache/e30_teacher_oof.npz")
    args = ap.parse_args()

    members = args.members.split(",")
    mean = None
    for m in members:
        D = np.load(f"analysis/cache/e30_oof_{m}.npz")
        assert D["covered"].all(), f"{m}: incomplete OOF coverage — run --merge first"
        p = D["probs"].astype(np.float64)
        mean = p if mean is None else mean + p
    mean /= len(members)

    # sanity: honest teachers must NOT be one-hot photocopies (the 1B failure signature)
    msp = mean.max(1)
    print(f"teacher MSP: mean {msp.mean():.3f} · p95 {np.quantile(msp, .95):.3f} "
          f"(1B in-sample teachers were ~0.99 everywhere — expect visibly softer here)")
    assert msp.mean() < 0.97, "teacher is near-one-hot — harvest looks in-sample, ABORT"

    np.savez(args.out, logits=np.log(np.clip(mean, 1e-9, 1.0)).astype(np.float32))
    print(f"teacher ({len(members)} members, {mean.shape[0]} rows) -> {args.out}")


if __name__ == "__main__":
    main()
