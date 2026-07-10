"""E27 — final flip rule: fit on FULL 14k val (user 2026-07-10), save artifact,
transfer-check on the deployment model's honest 3.5k slice.

Rule = list of (rank1_class, rank2_class, decile) cells where swapping the prediction
to rank-2 beats rank-1 on the fitting data (n >= min_n). Deciles are ALWAYS the
applying model's own MSP quantiles computed on the scored set (self-quantiles — no
fixed thresholds, robust to confidence-scale shift between models/sets).

Outputs:
  experiments/errorpred/flip_cells.json         the rule artifact (fitted on e9 val 14k)
  stdout: fitted cells + transfer eval on e8a_ls (deployment champion) 3.5k slice
          (e26_screen_logits.npz) + e9's own 3.5k as reference

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/flip_rule_final.py
"""
import json

import numpy as np

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, split_indices

SHORT = ["read", "grep", "lsdir", "glob", "edit", "write", "patch",
         "bash", "tests", "lint", "ask", "plan", "web", "resp"]
MIN_N = 20
OUT = "experiments/errorpred/flip_cells.json"


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def prep(z):
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    order = np.argsort(-z, 1)
    qs = np.quantile(msp, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(qs, msp, side="right") - 1, 0, 9)
    return order[:, 0], order[:, 1], dec


def apply_rule(cells, r1, r2, dec):
    pred = r1.copy()
    for a, b, d in cells:
        pred[(r1 == a) & (r2 == b) & (dec == d)] = b
    return pred


def main():
    D = np.load("analysis/cache/clpvi_tiers_granite_ls.npz")
    z9, yv = D["logits"], D["labels"]

    # ---- fit on the FULL 14k val ----
    r1, r2, dec = prep(z9)
    cells = []
    print("## final flip cells (fitted on e9_granite_ls, full 14k val, n>=20)\n")
    print("| rank1 | rank2 | decile | n | P(true=r1) | P(true=r2) |")
    print("|---|---|---|---|---|---|")
    for a in range(14):
        for b in range(14):
            for d in range(10):
                m = (r1 == a) & (r2 == b) & (dec == d)
                if m.sum() < MIN_N:
                    continue
                p1, p2 = (yv[m] == a).mean(), (yv[m] == b).mean()
                if p2 > p1:
                    cells.append((a, b, d))
                    print(f"| {SHORT[a]} | {SHORT[b]} | {d+1} | {int(m.sum())} | "
                          f"{p1:.3f} | {p2:.3f} |")
    with open(OUT, "w") as f:
        json.dump({"min_n": MIN_N, "fitted_on": "e9_granite_ls val14k",
                   "classes": ALL_CLASSES,
                   "cells": [{"rank1": SHORT[a], "rank2": SHORT[b], "decile": d}
                             for a, b, d in cells]}, f, indent=1)
    sel = apply_rule(cells, r1, r2, dec)
    print(f"\n{len(cells)} cells -> {OUT}")
    print(f"e9 val (IN-SAMPLE for the rule): acc {(r1 == yv).mean():.4f} -> "
          f"{(sel == yv).mean():.4f} · mF1 {mf1(yv, r1):.4f} -> {mf1(yv, sel):.4f}")

    # ---- transfer: deployment champion e8a_ls on its honest 3.5k slice ----
    samples, labels = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    from sklearn.model_selection import train_test_split
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    y35 = y_ids[va_eval]
    E = np.load("analysis/cache/e26_screen_logits.npz")

    print("\n## transfer check on the 3.5k full_data held-out slice (rule fixed, self-quantile deciles)\n")
    print("| model | base acc | rule acc | base mF1 | rule mF1 | ΔmF1 | rows swapped |")
    print("|---|---|---|---|---|---|---|")
    for tag in ["e8a_ls_richargs_full", "e9_granite_ls", "e8b_ls_qwen3_richargs_full"]:
        z = E[tag]
        r1t, r2t, dect = prep(z)
        selt = apply_rule(cells, r1t, r2t, dect)
        print(f"| {tag} | {(r1t == y35).mean():.4f} | {(selt == y35).mean():.4f} | "
              f"{mf1(y35, r1t):.4f} | {mf1(y35, selt):.4f} | "
              f"{mf1(y35, selt) - mf1(y35, r1t):+.4f} | {(selt != r1t).sum()} |")
    print("\nNOTE: e9 row overlaps the rule's fitting rows (14k val ⊃ 3.5k) — reference only;"
          "\ne8a_ls / e8b_ls rows are honest (models never saw them, rule never fitted on their logits).")


if __name__ == "__main__":
    main()
