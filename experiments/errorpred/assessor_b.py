"""E27 arm B — trained assessor: predict "granite-LS will get this row wrong" from
label-free features (user GO 2026-07-10). CPU-only, cached logits, no embeddings
(kNN-difficulty block deferred — needs a GPU embedding pass).

Feature blocks (all computable at test time):
  OWN    e9 logit shape: sorted top-4 softmax probs, MSP, entropy, top1-top2 margin,
         in-group margin, rank-1 identity (one-hot), rank-2 identity (one-hot)
  CROSS  per helper model (base/pvi06/aum06/cl/cart15/qwen3): its MSP, agreement with
         e9's r1, KL(e9 softmax ‖ helper softmax); plus total agree count

Target: e9 pred != gold. Eval: 5-fold CV AUROC, vs MSP-alone baseline; feature-block
ablation (OWN only vs OWN+CROSS). Model: sklearn HistGradientBoosting (xgboost absent).

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/assessor_b.py
"""
import numpy as np

from src.data import ALL_CLASSES, GROUP_ID

GID = np.array(GROUP_ID)


def softmax(z):
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def own_features(z):
    p = softmax(z)
    srt = np.sort(p, 1)[:, ::-1]
    r1 = z.argmax(1)
    zg = z.copy()
    zg[GID[None, :] != GID[r1][:, None]] = -np.inf
    zg[np.arange(len(z)), r1] = -np.inf
    pg = np.take_along_axis(p, zg.argmax(1)[:, None], 1)[:, 0]     # best in-group other
    ent = -(p * np.log(p + 1e-12)).sum(1)
    r2 = np.argsort(-z, 1)[:, 1]
    oh1 = np.eye(14)[r1]
    oh2 = np.eye(14)[r2]
    return np.column_stack([srt[:, :4], srt[:, 0], ent, srt[:, 0] - srt[:, 1],
                            srt[:, 0] - pg, oh1, oh2]), p, r1


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    D = np.load("analysis/cache/clpvi_tiers_granite_ls.npz")
    z9, y = D["logits"], D["labels"]
    X_own, p9, r1 = own_features(z9)
    err = (r1 != y).astype(int)

    helpers = dict(np.load("analysis/cache/coreset_gate_logits.npz"))
    helpers["qwen3"] = np.load("analysis/cache/qwen3_val_logits.npz")["logits"]
    cross = []
    agree_cnt = np.zeros(len(y))
    for name, zh in helpers.items():
        ph = softmax(zh)
        agree = (zh.argmax(1) == r1).astype(float)
        agree_cnt += agree
        kl = (p9 * (np.log(p9 + 1e-12) - np.log(ph + 1e-12))).sum(1)
        cross += [ph.max(1), agree, kl]
    X_cross = np.column_stack(cross + [agree_cnt])

    msp = p9.max(1)
    sets = {"MSP alone": -msp[:, None],
            "OWN logit-shape": X_own,
            "OWN + CROSS": np.column_stack([X_own, X_cross])}

    print("## E27 arm B · assessor 5-fold CV AUROC (target: e9 granite-LS error)\n")
    print(f"base error rate {err.mean():.3f} · n={len(y)}\n")
    print("| feature set | dims | AUROC (mean ± sd over folds) |")
    print("|---|---|---|")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_score = {}
    for name, X in sets.items():
        if X.shape[1] == 1:                                # msp needs no fitting
            aucs = [roc_auc_score(err[te], X[te, 0]) for _, te in skf.split(X, err)]
            oof = X[:, 0]
        else:
            oof = np.zeros(len(y))
            aucs = []
            for tr, te in skf.split(X, err):
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                   random_state=42)
                m.fit(X[tr], err[tr])
                s = m.predict_proba(X[te])[:, 1]
                oof[te] = s
                aucs.append(roc_auc_score(err[te], s))
        oof_score[name] = oof
        print(f"| {name} | {X.shape[1]} | {np.mean(aucs):.4f} ± {np.std(aucs):.4f} |")

    # actionability: error capture at fixed flag budgets, best set vs MSP
    best = oof_score["OWN + CROSS"]
    print("\n| flag budget | MSP: errors caught | assessor: errors caught |")
    print("|---|---|---|")
    for frac in (0.10, 0.20, 0.25):
        k = int(len(y) * frac)
        for_msp = err[np.argsort(msp)[:k]].sum()             # lowest msp first
        for_ass = err[np.argsort(best)[::-1][:k]].sum()
        print(f"| {frac:.0%} | {for_msp / err.sum():.1%} | {for_ass / err.sum():.1%} |")


if __name__ == "__main__":
    main()
