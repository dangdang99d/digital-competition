"""E29 — learned head over the trio, trained/evaluated on the E30 OOF caches.

Arm ① (GATEKEEPER, default): multinomial L2 logistic regression on the concatenated member
OOF probability vectors (3×14 = 42 dims). Arm ② (--arm emb, run only if ① clears the gate):
linear head on concatenated penultimate embeddings (3×768 = 2304 dims).

Evaluation: 5-fold CV reusing the E30 SESSION-GROUPED folds (StratifiedGroupKFold, seed 42 —
identical construction to the harvest). On a held-out fold, member probs came from fold-models
that never saw those sessions AND the head never trained on those rows → fully honest.
Null = uniform mean of member probs (the shipping combiner, LB 0.78719). Gate: mean ΔmF1
≥ +0.002 over the null, else E29 closes.

Self-contained data loading (train.jsonl order + train_labels.csv), replicating
src.data.load_samples exactly — runs on boxes without the full dacon env.

  python experiments/ensemble/e29_head.py                 # arm ①
  python experiments/ensemble/e29_head.py --arm emb       # arm ② (gated)
  python experiments/ensemble/e29_head.py --arm gate      # arm ③ per-row gating (gated)
"""
import argparse
import csv
import json
import re

import numpy as np

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
MEMBERS = ["is3", "aum06", "e25c"]
GATE = 0.002


def load_labels_and_sessions():
    """Row order = train.jsonl line order (== src.data.load_samples == the OOF cache order)."""
    ids = []
    with open("data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["id"])
    with open("data/train_labels.csv", encoding="utf-8") as f:
        lab = {r["id"]: r["action"] for r in csv.DictReader(f)}
    cid = {c: i for i, c in enumerate(ALL_CLASSES)}
    y = np.array([cid[lab[i]] for i in ids])
    sessions = np.array([re.sub(r"-step_\d+$", "", i) for i in ids])
    return y, sessions


def folds(y, sessions, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups=sessions))


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["probs", "emb", "gate"], default="probs")
    ap.add_argument("--C", type=float, nargs="+", default=[0.25, 1.0, 4.0])
    ap.add_argument("--balanced", action="store_true",
                    help="class_weight='balanced' — macro-F1-oriented objective (user 2026-07-12)")
    args = ap.parse_args()

    caches = {m: np.load(f"analysis/cache/e30_oof_{m}.npz") for m in MEMBERS}
    P = [caches[m]["probs"].astype(np.float64) for m in MEMBERS]
    y, sessions = load_labels_and_sessions()
    assert len(y) == P[0].shape[0], "row-order/coverage mismatch vs OOF caches"

    uni = sum(P) / len(P)
    null_pred = uni.argmax(1)
    print(f"n={len(y)} · uniform-mean null (all-OOF): mF1 {mf1(y, null_pred):.4f} · "
          f"acc {(null_pred == y).mean():.4f}")
    print("members solo (OOF): " + " · ".join(
        f"{m} {mf1(y, P[i].argmax(1)):.4f}" for i, m in enumerate(MEMBERS)))

    if args.arm == "probs":
        X = np.concatenate(P, 1)                                   # 70k x 42
    elif args.arm == "emb":
        X = np.concatenate([caches[m]["emb"].astype(np.float32) for m in MEMBERS], 1)  # 70k x 2304
    else:
        raise SystemExit("arm 'gate' not implemented until ② read")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fold_list = folds(y, sessions)
    print(f"\narm={args.arm} balanced={args.balanced} · X {X.shape} · 5-fold session-grouped CV vs uniform null\n")
    print("| C | head mF1 (mean over folds) | null mF1 | Δ | folds Δ>0 |")
    print("|---|---|---|---|---|")
    best = (-1, None)
    for C in args.C:
        hs, ns = [], []
        for tr, te in fold_list:
            clf = LogisticRegression(C=C, max_iter=2000,
                                     class_weight="balanced" if args.balanced else None)
            if args.arm == "emb":                       # raw penultimate features need scaling
                clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X[tr], y[tr])
            hs.append(mf1(y[te], clf.predict(X[te])))
            ns.append(mf1(y[te], null_pred[te]))
        d = np.array(hs) - np.array(ns)
        print(f"| {C} | {np.mean(hs):.4f} | {np.mean(ns):.4f} | {np.mean(d):+.4f} | "
              f"{(d > 0).sum()}/5 |")
        if np.mean(d) > best[0]:
            best = (np.mean(d), C)
    verdict = "PASS" if best[0] >= GATE else "FAIL"
    print(f"\nGATE (≥ +{GATE}): best Δ {best[0]:+.4f} at C={best[1]} → **{verdict}**")


if __name__ == "__main__":
    main()
