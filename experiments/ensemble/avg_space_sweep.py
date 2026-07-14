"""Averaging-space sweep over the 3.5k logit pool (CPU, free).

Question (user 2026-07-14): does uniform combining in a different SPACE beat our
shipped prob-mean? Spaces: prob-mean · logit-mean (argmax-identical to geometric
prob mean) · rank-mean · prob-median · majority vote. No learned weights (E29).

Combos scored: reigning SOTA trio (e28_full_t019/t023/t043, LB 0.78780), that trio
+ e38_t031fd, an all-E38-AWP set, and per-space greedy-forward (cap 4) over the
whole ok-pool. ⚠️ 3.5k slice screens only — it under-predicts and can mis-rank
ensembles (E8/E26); LB judges. Greedy picks here are candidates, not verdicts.
"""
import json
import os

import numpy as np
from loguru import logger
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

from src.data import CLASS_TO_ID, load_samples, split_indices
from src.runlog import log_cmd

DIR = "experiments/ensemble/logits"
C = 14


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(C), average="macro", zero_division=0)


def combine(logit_list, space):
    L = np.stack(logit_list)                      # (m, n, C)
    if space == "prob_mean":
        return softmax(L).mean(0).argmax(-1)
    if space == "logit_mean":                     # == geometric prob mean
        return L.mean(0).argmax(-1)
    if space == "rank_mean":                      # per-row rank of each class, avg over members
        R = np.stack([rankdata(l, axis=-1) for l in L])
        return R.mean(0).argmax(-1)
    if space == "prob_median":
        return np.median(softmax(L), 0).argmax(-1)
    if space == "vote":                           # majority vote, prob-mean tiebreak
        P = softmax(L)
        votes = np.zeros((L.shape[1], C))
        for m in range(L.shape[0]):
            votes[np.arange(L.shape[1]), P[m].argmax(-1)] += 1
        return np.where(votes.max(-1, keepdims=False) > L.shape[0] // 2,
                        votes.argmax(-1), P.mean(0).argmax(-1))
    raise ValueError(space)


def main():
    log_cmd()
    samples, labels = load_samples("./data")
    yid = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=yid[va], random_state=42)
    y = yid[va_eval]

    tags = {}
    with open(os.path.join(DIR, "manifest.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("decision") == "ok" and os.path.exists(os.path.join(DIR, f"{r['tag']}.npz")):
                tags[r["tag"]] = None
    pool = {t: np.load(os.path.join(DIR, f"{t}.npz"))["logits"] for t in tags}
    logger.info(f"pool: {len(pool)} models")
    solo = {t: mf1(y, l.argmax(-1)) for t, l in pool.items()}
    best_single = max(solo, key=solo.get)
    logger.info(f"best single: {best_single} {solo[best_single]:.4f}")

    SPACES = ["prob_mean", "logit_mean", "rank_mean", "prob_median", "vote"]
    named = {
        "SOTA trio (t019+t023+t043)": ["e28_full_t019", "e28_full_t023", "e28_full_t043"],
        "SOTA trio + t031fd": ["e28_full_t019", "e28_full_t023", "e28_full_t043", "e38_t031fd"],
        "E38 quad (t001+t031+t040+t070 fd)": ["e38_t001fd", "e38_t031fd", "e38_t040fd", "e38_t070fd"],
        "t031fd + t070fd + t019": ["e38_t031fd", "e38_t070fd", "e28_full_t019"],
    }
    print(f"\n{'combo':38s} " + " ".join(f"{s:>11s}" for s in SPACES))
    for name, members in named.items():
        if any(m not in pool for m in members):
            missing = [m for m in members if m not in pool]
            logger.warning(f"{name}: missing {missing}"); continue
        ls = [pool[m] for m in members]
        scores = [mf1(y, combine(ls, s)) for s in SPACES]
        print(f"{name:38s} " + " ".join(f"{v:11.4f}" for v in scores))

    # greedy-forward with replacement, cap 4, per space (slice-screen; LB judges)
    print()
    for space in SPACES:
        sel = [best_single]
        cur = mf1(y, combine([pool[t] for t in sel], space))
        while len(sel) < 4:
            cands = [(mf1(y, combine([pool[t] for t in sel + [c]], space)), c) for c in pool]
            gain, pick = max(cands)
            if gain <= cur + 1e-6:
                break
            sel.append(pick); cur = gain
        print(f"greedy[{space:11s}] F1={cur:.4f}  members={sel}")


if __name__ == "__main__":
    main()
