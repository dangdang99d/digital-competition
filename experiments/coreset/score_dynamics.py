"""C2/C4/C5/C6 — dynamics-based drop-noisy scorers.

Reads a per-epoch train-dynamics npz (from `finetune.py --log_dynamics`: probs (E,N,C),
tr, labels) and emits a keep-set of ABSOLUTE indices per method (drop the noisy tail).
All four scores come from the ONE instrumented baseline run — they differ only in the
statistic over training dynamics:

  C4 Cartography (Swayamdipta EMNLP'20): confidence = mean_e p_e[true];
     variability = std_e p_e[true]. easy=hi-conf/lo-var, ambiguous=hi-var,
     hard=lo-conf/lo-var (≈ mislabeled). Drop-noisy = drop HARD, keep easy+ambiguous.
  C2 AUM (Pleiss NeurIPS'20): margin = mean_e (p_true - max_other). Low/neg = mislabeled.
  C5 Forgetting (Toneva ICLR'19): never-learned (correct in 0 epochs) ≈ unlearnable.
  C6 EL2N (Paul NeurIPS'21): ||p - onehot(true)||_2 at an EARLY epoch. High = hard.

Usage:
  python experiments/coreset/score_dynamics.py --dyn <dynamics.npz> --drop 0.06 0.15
Keep-sets -> experiments/coreset/keepsets/<method>_drop<pct>.npy ; data map -> figures/.
"""
import argparse
import os

import numpy as np

KEEPDIR = "experiments/coreset/keepsets"
FIGDIR = "experiments/coreset/figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn", required=True, help="dynamics npz from finetune --log_dynamics")
    ap.add_argument("--drop", type=float, nargs="+", default=[0.06, 0.15],
                    help="fractions of train to drop as noisy (per method)")
    args = ap.parse_args()
    os.makedirs(KEEPDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)

    D = np.load(args.dyn)
    P, tr, y = D["probs"], D["tr"], D["labels"]      # P:(E,N,C)  tr,y:(N,)
    E, N, C = P.shape
    idx = np.arange(N)
    ptrue = P[:, idx, y]                               # (E,N) prob of the gold label per epoch
    pred = P.argmax(2)                                 # (E,N)
    correct = (pred == y[None, :])                     # (E,N)

    # --- statistics ---
    conf = ptrue.mean(0)                               # cartography x-axis
    var = ptrue.std(0)                                 # cartography y-axis
    corr_rate = correct.mean(0)
    # AUM margin (prob-space proxy of logit margin): p_true - max_other, mean over epochs
    P2 = P.copy(); P2[:, idx, y] = -1
    aum = (ptrue - P2.max(2)).mean(0)
    # forgetting: correct->wrong transitions; never-learned = correct in 0 epochs
    never = correct.sum(0) == 0
    forget = np.maximum(0, (correct[:-1].astype(int) - correct[1:].astype(int))).sum(0)
    # EL2N at the earliest epoch (least memorized)
    onehot = np.eye(C)[y]
    el2n = np.linalg.norm(P[0] - onehot, axis=1)

    print(f"dynamics {args.dyn}: E={E} N={N} C={C}")
    print(f"  conf med {np.median(conf):.3f} | never-learned {never.mean():.2%} | "
          f"AUM<0 {np.mean(aum < 0):.2%}")

    def emit(name, drop_score, frac, keep_extra=None):
        """Drop the `frac` lowest-'goodness' samples (drop_score high = noisier)."""
        k = int(frac * N)
        order = np.argsort(-drop_score)               # noisiest first
        drop = set(order[:k].tolist())
        if keep_extra is not None:                    # never drop these (e.g. ambiguous)
            drop -= set(np.where(keep_extra)[0].tolist())
        keep_mask = np.array([i not in drop for i in range(N)])
        keep_abs = tr[keep_mask]
        out = f"{KEEPDIR}/{name}_drop{int(frac*100):02d}.npy"
        np.save(out, keep_abs)
        print(f"  {name:22s} drop {frac:.0%}: kept {len(keep_abs):5d}/{N} -> {out}")

    for f in args.drop:
        # C4 cartography: drop HARD (lo-conf & lo-var); protect AMBIGUOUS (hi-var)
        ambiguous = var > np.quantile(var, 0.66)
        hardness = (1 - conf) * (var < np.median(var))     # hard = low conf AND low variability
        emit("cart_drophard", hardness, f, keep_extra=ambiguous)
        # C2 AUM: drop lowest margin
        emit("aum", -aum, f)
        # C6 EL2N: drop highest error-norm
        emit("el2n", el2n, f)
    # C5 forgetting: drop the never-learned (a fixed set, not a fraction)
    keep_abs = tr[~never]
    np.save(f"{KEEPDIR}/forget_neverlearned.npy", keep_abs)
    print(f"  forget_neverlearned    drop {never.mean():.1%}: kept {len(keep_abs)}/{N}")

    _datamap(conf, var, corr_rate)


def _datamap(conf, var, corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG = "#fcfcfb"
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    sc = ax.scatter(var, conf, c=corr, cmap="RdYlBu", s=4, alpha=0.5, vmin=0, vmax=1)
    ax.set_xlabel("variability (std of gold-label prob)")
    ax.set_ylabel("confidence (mean gold-label prob)")
    ax.set_title("Data map — easy (top-left) / ambiguous (right) / hard (bottom-left)")
    fig.colorbar(sc, label="correctness")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/datamap.png", dpi=130, facecolor=BG)
    print(f"  data map -> {FIGDIR}/datamap.png")


if __name__ == "__main__":
    main()
