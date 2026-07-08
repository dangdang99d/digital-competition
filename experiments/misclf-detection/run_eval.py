"""M1/M2/M3/M5 misclassification detectors, evaluated on the frozen-qwen3 substrate.

All detectors output a CONFIDENCE score (higher = model more likely CORRECT). We rank
correctness with it and report, per detector:
  AUROC(correct vs wrong) overall + per slice (sim / au / first-step / later) · AURC.
Fit-vs-eval discipline: M2 stats and M3's MLP are fit on the TRAIN split only; every
AUROC/AURC number is on the held-out 14k val. M1/M5 have nothing to fit.

Baseline to beat: single-model margin AUROC ~0.83 (this run reprints it as M1:margin).

Run:
  PYTHONPATH=/home/ocean/dacon CUDA_VISIBLE_DEVICES=3 \
    /home/ocean/miniconda3/envs/dacon/bin/python experiments/misclf-detection/run_eval.py
"""
import json
import os

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

SUB = "analysis/cache/misclf_substrate_qwen3.npz"
HIST0 = "analysis/cache/hist0_val_logits.npz"       # 2nd frozen model for M5 (bge-m3), va-order
FIGDIR = "experiments/misclf-detection/figures"
OUTJSON = "experiments/misclf-detection/m_results.json"
NC = 14


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logsumexp(z):
    m = z.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(z - m).sum(1)))


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def aurc(conf, correct):
    """Area under the risk-coverage curve (lower = better)."""
    order = np.argsort(-conf, kind="stable")
    cs = correct[order].astype(float)
    n = len(cs)
    risk = 1 - np.cumsum(cs) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    return float(np.trapz(risk, cov))


def risk_coverage(conf, correct):
    order = np.argsort(-conf, kind="stable")
    cs = correct[order].astype(float)
    n = len(cs)
    return np.arange(1, n + 1) / n, 1 - np.cumsum(cs) / np.arange(1, n + 1)


def load_data(sub):
    """Full substrate if present; otherwise a logits-only fallback from the cached
    original-champion val logits — enough for M1/M5."""
    from src.data import CLASS_TO_ID, load_samples, split_indices
    if os.path.exists(sub):
        S = np.load(sub, allow_pickle=True)
        lg, ft, lb = S["logits"], S["feats"].astype(np.float32), S["labels"]
        tr, va = S["tr"], S["va"]
        return dict(zval=lg[va], yval=lb[va], gval=S["gen"][va], sval=S["step"][va], va=va,
                    ftr=ft[tr], ytr=lb[tr], ztr=lg[tr], fva=ft[va], src=f"substrate {sub}")
    import re
    _ID = re.compile(r"sess_([a-z]+)_.*-step_(\d+)")
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    zval = np.load("analysis/cache/qwen3_val_logits.npz")["logits"]
    return dict(zval=zval, va=va,
                yval=np.array([CLASS_TO_ID[y[i]] for i in va]),
                gval=np.array([_ID.match(samples[i]["id"]).group(1) for i in va]),
                sval=np.array([int(_ID.match(samples[i]["id"]).group(2)) for i in va]),
                ftr=None, ytr=None, fva=None, src="cache (logits-only — M2/M3 skipped)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=SUB)
    ap.add_argument("--heldout", action="store_true",
                    help="restrict val to the full_data clean 25%% held-out (never seen by a full_data model)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    D = load_data(args.sub)
    logger.info(f"data source: {D['src']}")
    zval, yval, gval, sval = D["zval"], D["yval"], D["gval"], D["sval"]

    hmask = None
    if args.heldout:  # the 25% of the seed-42 val that a --full_data model never trained on
        from sklearn.model_selection import train_test_split
        from src.data import CLASS_TO_ID, load_samples
        _, y = load_samples("./data")
        yid = np.array([CLASS_TO_ID[a] for a in y])
        va = D["va"]
        _, va_eval = train_test_split(va, test_size=0.25, stratify=yid[va], random_state=42)
        hmask = np.isin(va, va_eval)
        zval, yval, gval, sval = zval[hmask], yval[hmask], gval[hmask], sval[hmask]
        if D["fva"] is not None:
            D["fva"] = D["fva"][hmask]
        logger.info(f"heldout: {int(hmask.sum())} clean samples of {len(hmask)} val")

    pval = softmax(zval)
    pred = zval.argmax(1)
    correct = (pred == yval).astype(int)
    logger.info(f"val n={len(zval)} acc={correct.mean():.4f}  (wrong={int((1-correct).sum())})")

    slices = {"overall": np.ones(len(zval), bool),
              "sim": gval == "sim", "au": gval == "au",
              "first-step": sval == 1, "later": sval >= 2}

    scores = {}

    # ---------- M1 · output-based (free, cached logits) ----------
    s2 = np.sort(zval, 1)
    margin = s2[:, -1] - s2[:, -2]
    scores["M1:msp"] = pval.max(1)
    scores["M1:neg_entropy"] = (pval * np.log(pval + 1e-12)).sum(1)
    scores["M1:margin"] = margin
    scores["M1:max_logit"] = zval.max(1)
    scores["M1:energy"] = logsumexp(zval)
    scores["M1:doctor"] = (pval ** 2).sum(1)

    # ---------- M5 · cross-model disagreement (free, base qwen3 + hist0) ----------
    have_hist0 = os.path.exists(HIST0)
    if have_hist0:
        zh = np.load(HIST0)["logits"]
        if hmask is not None:
            zh = zh[hmask]
        assert zh.shape[0] == len(zval), "hist0 cache not aligned to val"
        predh = zh.argmax(1)
        mh = np.sort(zh, 1)
        marginh = mh[:, -1] - mh[:, -2]
        agree = (pred == predh).astype(float)
        scores["M5:agree"] = agree
        scores["M5:sum_norm_margin"] = zscore(margin) + zscore(marginh)
        scores["M5:agree_gated_margin"] = margin - 1e6 * (1 - agree)  # disagreements ranked lowest
    else:
        logger.warning("hist0 cache missing — M5 skipped")

    # ---------- M2 · distance/density + M3 · ConfidNet (feature-based) ----------
    if D["fva"] is None:
        logger.warning("logits-only substrate — M2/M3 (feature-based) skipped")
    else:
        ftr, ytr, ztr, fva = D["ftr"], D["ytr"], D["ztr"], D["fva"]
        H = ftr.shape[1]
        nval = len(fva)
        # M2 Mahalanobis, tied covariance (Lee et al. 2018)
        mus = np.stack([ftr[ytr == c].mean(0) for c in range(NC)])
        Xc = ftr - mus[ytr]
        prec = np.linalg.inv(np.cov(Xc.T) + 1e-3 * np.eye(H)).astype(np.float32)
        maha = np.empty((nval, NC), dtype=np.float32)
        for c in range(NC):
            diff = fva - mus[c]
            maha[:, c] = np.einsum("ij,jk,ik->i", diff, prec, diff, optimize=True)
        scores["M2:mahalanobis"] = -maha.min(1)

        def l2n(x):
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        try:
            from sklearn.neighbors import NearestNeighbors
            _, nbr = NearestNeighbors(n_neighbors=10).fit(l2n(ftr)).kneighbors(l2n(fva))
            scores["M2:knn_agree"] = (ytr[nbr] == pred[:, None]).mean(1)
            per_class_nn = {c: NearestNeighbors(n_neighbors=1).fit(l2n(ftr[ytr == c]))
                            for c in range(NC)}
            fv = l2n(fva)
            d_all = np.stack([per_class_nn[c].kneighbors(fv)[0][:, 0] for c in range(NC)], 1)
            d_pred = d_all[np.arange(nval), pred]
            d_all[np.arange(nval), pred] = np.inf
            scores["M2:trustscore"] = d_all.min(1) / (d_pred + 1e-9)  # provisional vs google/TrustScore
        except Exception as e:
            logger.warning(f"kNN/TrustScore skipped: {e}")

        # M3 ConfidNet: small MLP on frozen feats, regress TCP = p_frozen[true]
        try:
            import torch
            import torch.nn as nn
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            tcp = softmax(ztr)[np.arange(len(ytr)), ytr].astype(np.float32)
            Xtr, Ytr = torch.tensor(ftr), torch.tensor(tcp)
            net = nn.Sequential(nn.Linear(H, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
            lossf, g = nn.MSELoss(), torch.Generator().manual_seed(0)
            net.train()
            for ep in range(25):
                perm = torch.randperm(len(Xtr), generator=g)
                for i in range(0, len(Xtr), 1024):
                    b = perm[i:i + 1024]
                    opt.zero_grad()
                    o = torch.sigmoid(net(Xtr[b].to(dev))).squeeze(1)
                    loss = lossf(o, Ytr[b].to(dev)); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                scores["M3:confidnet"] = torch.sigmoid(
                    net(torch.tensor(fva).to(dev))).squeeze(1).cpu().numpy()
            logger.info(f"ConfidNet trained (last MSE {loss.item():.4f})")
        except Exception as e:
            logger.warning(f"ConfidNet skipped: {e}")

    # ---------- evaluate everything ----------
    rows = []
    for name, conf in scores.items():
        row = {"detector": name, "aurc": aurc(conf, correct)}
        for sl, m in slices.items():
            c = correct[m]
            row[sl] = float("nan") if c.min() == c.max() else roc_auc_score(c, conf[m])
        rows.append(row)
    rows.sort(key=lambda r: -r["overall"])

    hdr = f"{'detector':24s} {'overall':>8} {'sim':>7} {'au':>7} {'first':>7} {'later':>7} {'AURC':>8}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['detector']:24s} {r['overall']:8.4f} {r['sim']:7.4f} {r['au']:7.4f} "
              f"{r['first-step']:7.4f} {r['later']:7.4f} {r['aurc']:8.4f}")
    print(f"\nbaseline (M1:margin) overall AUROC = "
          f"{next(r['overall'] for r in rows if r['detector']=='M1:margin'):.4f}")

    suff = ("_" + args.tag) if args.tag else ""
    outjson = OUTJSON.replace(".json", f"{suff}.json")
    os.makedirs(FIGDIR, exist_ok=True)
    json.dump({"val_acc": float(correct.mean()), "n_val": int(len(zval)),
               "heldout": bool(args.heldout), "sub": args.sub, "rows": rows},
              open(outjson, "w"), indent=2)
    plot(rows, scores, correct, slices, suff)
    logger.success(f"results -> {outjson}")


def plot(rows, scores, correct, slices, suff=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, BLUE, RED, GREY = "#fcfcfb", "#3987e5", "#d4553f", "#b8b8b3"

    # fig 1: overall AUROC bars, baseline line
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    names = [r["detector"] for r in rows]
    vals = [r["overall"] for r in rows]
    base = next(r["overall"] for r in rows if r["detector"] == "M1:margin")
    cols = [BLUE if v >= base else GREY for v in vals]
    ax.barh(range(len(names)), vals, color=cols)
    ax.axvline(base, color=RED, lw=1.4, ls="--", label=f"margin baseline {base:.3f}")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0.5, max(vals) + 0.03)
    ax.set_xlabel("AUROC (correct vs wrong), 14k val"); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Misclassification detectors — overall ranking (frozen qwen3)", fontsize=11)
    ax.grid(axis="x", color=GREY, alpha=0.3); [s.set_visible(False) for s in ax.spines.values()]
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/m_auroc_overall{suff}.png", dpi=130, facecolor=BG)

    # fig 2: per-slice robustness for top-6 detectors
    top = rows[:6]
    sln = ["sim", "au", "first-step", "later"]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG); ax.set_facecolor(BG)
    x = np.arange(len(sln)); w = 0.8 / len(top)
    for i, r in enumerate(top):
        ax.bar(x + i * w, [r[s] for s in sln], w, label=r["detector"])
    ax.axhline(0.83, color=RED, lw=1.2, ls="--", alpha=0.7)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(sln)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Robustness by slice (top-6) — must hold on au + first-step (E6 lesson)", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", color=GREY, alpha=0.3)
    [s.set_visible(False) for s in ax.spines.values()]
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/m_robustness_slices{suff}.png", dpi=130, facecolor=BG)

    # fig 3: risk-coverage curves, top-5
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG); ax.set_facecolor(BG)
    for r in rows[:5]:
        cov, risk = risk_coverage(scores[r["detector"]], correct)
        ax.plot(cov, risk, lw=1.6, label=f"{r['detector']} (AURC {r['aurc']:.3f})")
    ax.set_xlabel("coverage"); ax.set_ylabel("risk (error rate on covered)")
    ax.set_title("Risk–coverage (lower = better), frozen qwen3", fontsize=11)
    ax.legend(fontsize=8); ax.grid(color=GREY, alpha=0.3)
    [s.set_visible(False) for s in ax.spines.values()]
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/m_risk_coverage{suff}.png", dpi=130, facecolor=BG)
    logger.info(f"figures -> {FIGDIR}/m_*.png")


if __name__ == "__main__":
    main()
