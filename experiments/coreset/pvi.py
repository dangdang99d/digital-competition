"""C7 — Pointwise V-Information (Ethayarajh et al., ICML'22) drop-noisy scorer.

PVI(x→y) = log2 g[x](y) − log2 g'[∅](y):
  g[x](y)   = out-of-sample prob of the gold label (reuse the cleanlab k-fold OOF).
  g'[∅](y)  = a null-input model's prob of the gold label = the class prior (training on
              empty input converges to the marginal; standard approximation, no extra run).
Low/negative PVI = the input carries no usable information for that sample ≈ unlearnable /
mislabeled. Drop-noisy = drop the lowest-PVI tail. Free: piggybacks on C1's OOF npz.

  python experiments/coreset/pvi.py --oof analysis/cache/coreset_oof_granite.npz --drop 0.06 0.15
"""
import argparse
import os

import numpy as np

KEEPDIR = "experiments/coreset/keepsets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="analysis/cache/coreset_oof_granite.npz",
                    help="k-fold OOF npz from confident_learning.py (oof, tr, labels)")
    ap.add_argument("--drop", type=float, nargs="+", default=[0.06, 0.15])
    args = ap.parse_args()
    os.makedirs(KEEPDIR, exist_ok=True)

    D = np.load(args.oof)
    oof, tr, y = D["oof"], D["tr"], D["labels"]
    N = len(tr)
    prior = np.bincount(y, minlength=oof.shape[1]) / N        # g'[∅](y) ≈ class prior
    idx = np.arange(N)
    g_x = np.clip(oof[idx, y], 1e-6, 1.0)
    pvi = np.log2(g_x) - np.log2(prior[y] + 1e-12)
    print(f"PVI: mean {pvi.mean():.3f}  |  PVI<0 (unlearnable) {np.mean(pvi < 0):.2%}")

    for f in args.drop:
        k = int(f * N)
        drop = np.argsort(pvi)[:k]                              # lowest PVI first
        keep_mask = np.ones(N, bool); keep_mask[drop] = False
        out = f"{KEEPDIR}/pvi_drop{int(f*100):02d}.npy"
        np.save(out, tr[keep_mask])
        print(f"  pvi drop {f:.0%}: kept {keep_mask.sum()}/{N} -> {out}")


if __name__ == "__main__":
    main()
