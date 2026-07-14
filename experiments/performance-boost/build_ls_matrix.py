"""E47: build a (14,14) soft-target matrix for `finetune.py --loss lsmat`.

Two modes:
  neg      uniform GENERALIZED label smoothing (Wei et al. ICML'22, 2106.04149):
           T = (1-eps)*I + eps/K, with eps allowed NEGATIVE (true class >1, all
           other classes get a small negative target — pushes wrong logits DOWN
           harder; the paper's fit for high-label-noise regimes).
  grouped  user-proposed structured smoothing (2026-07-14): confusion GROUPS get
           small POSITIVE mass (they're plausibly-true under our ~20% noise +
           ambiguity — E27: true label is in the top-1's confusion group >=95%),
           everything out-of-group gets NEGATIVE mass:
             t_y = 1 - eps_in*|G_y \\ y| + eps_out*|not G_y|
             t_j = +eps_in   (j in G_y, j != y)
             t_j = -eps_out  (j not in G_y)
           Rows sum to 1 by construction.

Groups come from the honest E30 OOF confusion structure (train-side signal only —
no test-side calibration): mean of the 3 members' OOF probs -> row-normalized
confusion C[true, pred] -> symmetrized off-diagonal affinity -> average-linkage
agglomerative clustering into --n_groups. Groups are printed for eyeball sanity.
"""
import argparse
import json

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples
from src.runlog import log_cmd

OOF = ["analysis/cache/e30_oof_is3.npz", "analysis/cache/e30_oof_aum06.npz",
       "analysis/cache/e30_oof_e25c.npz"]
K = 14


def confusion_groups(n_groups):
    _, labels = load_samples("./data")
    y = np.array([CLASS_TO_ID[a] for a in labels])
    p = np.mean([np.load(f)["probs"] for f in OOF], axis=0)
    C = np.zeros((K, K))
    for k in range(K):
        C[k] = p[y == k].mean(0)                       # soft confusion row (honest OOF)
    A = C / C.sum(1, keepdims=True)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = squareform((A.max() - A), checks=False)        # affinity -> distance
    lab = fcluster(linkage(D, method="average"), n_groups, criterion="maxclust")
    groups = [np.where(lab == g)[0].tolist() for g in sorted(set(lab))]
    for g in groups:
        logger.info("group: " + ", ".join(ALL_CLASSES[i] for i in g))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["neg", "grouped"])
    ap.add_argument("--eps", type=float, default=-0.1, help="neg mode: GLS eps (may be <0)")
    ap.add_argument("--eps_in", type=float, default=0.02,
                    help="grouped: PER-CLASS positive mass for in-group classes")
    ap.add_argument("--eps_out", type=float, default=0.01,
                    help="grouped: PER-CLASS negative mass for out-of-group classes")
    ap.add_argument("--n_groups", type=int, default=5)
    ap.add_argument("--out", required=True, help="output .npy path")
    args = ap.parse_args()
    log_cmd()

    if args.mode == "neg":
        T = (1.0 - args.eps) * np.eye(K) + args.eps / K
        meta = {"mode": "neg", "eps": args.eps}
    else:
        groups = confusion_groups(args.n_groups)
        gid = np.empty(K, dtype=int)
        for gi, g in enumerate(groups):
            for c in g:
                gid[c] = gi
        T = np.zeros((K, K))
        for yc in range(K):
            ing = [j for j in range(K) if gid[j] == gid[yc] and j != yc]
            outg = [j for j in range(K) if gid[j] != gid[yc]]
            T[yc, ing] = args.eps_in
            T[yc, outg] = -args.eps_out
            T[yc, yc] = 1.0 - args.eps_in * len(ing) + args.eps_out * len(outg)
        meta = {"mode": "grouped", "eps_in": args.eps_in, "eps_out": args.eps_out,
                "groups": [[ALL_CLASSES[i] for i in g] for g in groups]}

    assert np.allclose(T.sum(1), 1.0), "rows must sum to 1"
    np.save(args.out, T.astype(np.float32))
    with open(args.out.replace(".npy", ".json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    logger.success(f"{args.out}: diag [{T.diagonal().min():.3f},{T.diagonal().max():.3f}], "
                   f"offdiag [{(T - np.diag(T.diagonal())).min():.4f},"
                   f"{(T - np.diag(T.diagonal())).max():.4f}]")


if __name__ == "__main__":
    main()
