"""E27 — flip rule variant: rank-2 REDEFINED as best IN-GROUP runner-up (user 2026-07-10).

Rationale: group(r1) is right ~99% (§5), so out-of-group raw rank-2s are noise AND cost
coverage (rows whose raw r2 is out-of-group never match a flip cell even when the in-group
runner-up carries the signal). Here r2g = argmax logit over group(r1) \\ {r1}.

Same three evaluations as the raw-r2 pipeline, for direct comparison:
  1. cross-fitted swap policy (2-fold x 5 seeds) on all 7 cached models
  2. full-fit rule on e9 val 14k -> experiments/errorpred/flip_cells_group.json
  3. honest transfer to e8a_ls / e8b_ls on the 3.5k slice

  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/errorpred/variant_group_r2.py
"""
import json

import numpy as np

from src.data import ALL_CLASSES, CLASS_TO_ID, GROUP_ID, load_samples, split_indices

SHORT = ["read", "grep", "lsdir", "glob", "edit", "write", "patch",
         "bash", "tests", "lint", "ask", "plan", "web", "resp"]
MIN_N = 20
GID = np.array(GROUP_ID)


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def prep_group(z):
    """r1, in-group runner-up r2g, self-quantile deciles."""
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    msp = p.max(1)
    r1 = z.argmax(1)
    zg = z.copy()
    zg[GID[None, :] != GID[r1][:, None]] = -np.inf      # keep only group(r1) classes
    zg[np.arange(len(z)), r1] = -np.inf                 # drop r1 itself
    r2g = zg.argmax(1)
    qs = np.quantile(msp, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(qs, msp, side="right") - 1, 0, 9)
    return r1, r2g, dec


def fit_cells(y, r1, r2, dec, rows):
    cells = []
    key = (r1[rows] * 14 + r2[rows]) * 10 + dec[rows]
    for k in np.unique(key):
        m = rows[key == k]
        if len(m) < MIN_N:
            continue
        a, b = (k // 10) // 14, (k // 10) % 14
        if (y[m] == b).mean() > (y[m] == a).mean():
            cells.append((int(a), int(b), int(k % 10)))
    return cells


def apply_cells(cells, r1, r2, dec):
    pred = r1.copy()
    for a, b, d in cells:
        pred[(r1 == a) & (r2 == b) & (dec == d)] = b
    return pred


def main():
    D = np.load("analysis/cache/clpvi_tiers_granite_ls.npz")
    yv = D["labels"]
    models = {"granite_ls(e9)": D["logits"]}
    models.update(dict(np.load("analysis/cache/coreset_gate_logits.npz")))
    models["qwen3"] = np.load("analysis/cache/qwen3_val_logits.npz")["logits"]

    print("## group-restricted r2 — cross-fitted swap policy (2-fold × 5 seeds)\n")
    print("| model | Δacc mean | ΔmF1 mean [min,max] | cells/run |")
    print("|---|---|---|---|")
    for name, z in models.items():
        r1, r2, dec = prep_group(z)
        base_acc, base_f1 = (r1 == yv).mean(), mf1(yv, r1)
        da, df, nc = [], [], []
        for s in (42, 1, 2, 3, 4):
            rng = np.random.default_rng(s)
            perm = rng.permutation(len(yv))
            halves = [perm[: len(yv) // 2], perm[len(yv) // 2:]]
            pred = r1.copy()
            tot = 0
            for f in (0, 1):
                cells = fit_cells(yv, r1, r2, dec, halves[f])
                app = halves[1 - f]
                pred[app] = apply_cells(cells, r1[app], r2[app], dec[app])
                tot += len(cells)
            da.append((pred == yv).mean() - base_acc)
            df.append(mf1(yv, pred) - base_f1)
            nc.append(tot)
        da, df = np.array(da), np.array(df)
        print(f"| {name} | {da.mean():+.4f} | {df.mean():+.4f} [{df.min():+.4f},{df.max():+.4f}] "
              f"| {min(nc)}–{max(nc)} |")

    # ---- full fit on e9 + artifact ----
    r1, r2, dec = prep_group(models["granite_ls(e9)"])
    cells = fit_cells(yv, r1, r2, dec, np.arange(len(yv)))
    print(f"\n## full-fit cells (e9 val 14k, group-restricted r2): {len(cells)}\n")
    print("| rank1 | r2g | decile | n | P(true=r1) | P(true=r2g) |")
    print("|---|---|---|---|---|---|")
    for a, b, d in cells:
        m = (r1 == a) & (r2 == b) & (dec == d)
        print(f"| {SHORT[a]} | {SHORT[b]} | {d+1} | {int(m.sum())} | "
              f"{(yv[m] == a).mean():.3f} | {(yv[m] == b).mean():.3f} |")
    with open("experiments/errorpred/flip_cells_group.json", "w") as f:
        json.dump({"min_n": MIN_N, "fitted_on": "e9_granite_ls val14k",
                   "r2_definition": "best in-group runner-up", "classes": ALL_CLASSES,
                   "cells": [{"rank1": SHORT[a], "rank2": SHORT[b], "decile": d}
                             for a, b, d in cells]}, f, indent=1)
    sel = apply_cells(cells, r1, r2, dec)
    print(f"\ne9 val (rule IN-SAMPLE): mF1 {mf1(yv, r1):.4f} -> {mf1(yv, sel):.4f}")

    # ---- honest transfer: 3.5k slice ----
    samples, labels = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    from sklearn.model_selection import train_test_split
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    y35 = y_ids[va_eval]
    E = np.load("analysis/cache/e26_screen_logits.npz")
    print("\n## honest transfer, 3.5k slice (rule fixed from e9)\n")
    print("| model | base mF1 | rule mF1 | ΔmF1 | swapped |")
    print("|---|---|---|---|---|")
    for tag in ["e8a_ls_richargs_full", "e8b_ls_qwen3_richargs_full"]:
        r1t, r2t, dect = prep_group(E[tag])
        selt = apply_cells(cells, r1t, r2t, dect)
        print(f"| {tag} | {mf1(y35, r1t):.4f} | {mf1(y35, selt):.4f} | "
              f"{mf1(y35, selt) - mf1(y35, r1t):+.4f} | {(selt != r1t).sum()} |")


if __name__ == "__main__":
    main()
