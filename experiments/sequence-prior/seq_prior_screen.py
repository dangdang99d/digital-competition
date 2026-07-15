"""E48 — count-based history sequence prior (competitor-borrowed, youngyoung2345).

Idea: build a fixed categorical prior over the 14 actions keyed on the ordered
sequence of prior assistant-action NAMES in a row's history (optionally + the last
action's result status). Add it to the model log-probs at inference:

    argmax( log p_model(x) + weight * log_prior[key(x)] )

applied ONLY to `sess_sim_` rows with a non-empty action history and whose key has
enough training support; every other row (AU, history-0, under-supported) is kept
BIT-EXACT to the baseline argmax. This is NOT a learned combiner (those are dead —
[[uniform-mean-unbeatable]]); it is a non-neural structured prior.

Leak-free eval: the prior is built ONLY from the 56k fold-0 TRAIN rows; it is applied
to the 14k fold-0 VAL rows, scored against the SAME champion AWP model (t031) whose
fold-56k checkpoint never saw the 14k val. Baseline = argmax of the banked t031 softmax
probs in output/e38/fold_screen_probs.npz.

Run (CPU, seconds):
    PYTHONPATH=. python -u experiments/sequence-prior/seq_prior_screen.py
"""
import json
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import f1_score

from src.data import (ALL_CLASSES, CLASS_TO_ID, load_samples,
                      session_fold_indices)

FOLD, N_SPLITS, SEED = 0, 5, 42
FINDING_IDS = np.arange(4)  # read_file, grep_search, list_directory, glob_pattern
PROBS_NPZ = "output/e38/fold_screen_probs.npz"
BASE_TAG = "t031"  # champion AWP single (LB 0.79300)


# ----------------------------------------------------------------------------- keys
def session_of(sample_id):
    return re.sub(r"-step_\d+$", "", sample_id)


def is_sim(sample_id):
    return sample_id.startswith("sess_sim_")


def action_names(sample):
    """Ordered list of prior assistant-action names in the row's history."""
    return [t.get("name", "") for t in sample.get("history", [])
            if t.get("role") == "assistant_action"]


def result_status(sample):
    """Coarse status of the LAST history action's result_summary."""
    acts = [t for t in sample.get("history", []) if t.get("role") == "assistant_action"]
    if not acts:
        return "NONE"
    summ = str(acts[-1].get("result_summary", "")).lower()
    if not summ:
        return "NONE"
    if any(w in summ for w in ("error", "fail", "not found", "no such", "denied", "timeout")):
        return "ERR"
    return "OK"


def make_key(sample, mode, lastk):
    names = action_names(sample)
    if not names:
        return None
    seq = names[-lastk:] if lastk else names
    key = " ".join(seq)
    if mode == "status":
        key = key + " [R_" + result_status(sample) + "]"
    return key


# ----------------------------------------------------------------------------- prior
def build_prior(train_samples, train_yid, mode, lastk):
    """key -> (log_prior[14], n_support_sessions).  Add-1 (Laplace) smoothed."""
    counts = defaultdict(lambda: np.zeros(len(ALL_CLASSES), dtype=np.float64))
    sessions = defaultdict(set)
    for s, y in zip(train_samples, train_yid):
        if not is_sim(s["id"]):
            continue
        key = make_key(s, mode, lastk)
        if key is None:
            continue
        counts[key][y] += 1.0
        sessions[key].add(session_of(s["id"]))
    prior = {}
    for key, c in counts.items():
        logp = np.log((c + 1.0) / (c.sum() + len(ALL_CLASSES)))
        prior[key] = (logp, len(sessions[key]))
    return prior


# ----------------------------------------------------------------------------- apply
def apply_prior(base_logp, val_samples, prior, mode, lastk, weight, min_support,
                seen_only=False):
    """Return (adjusted_pred, applied_mask). Non-applied rows keep baseline argmax."""
    pred = base_logp.argmax(1).copy()
    applied = np.zeros(len(val_samples), dtype=bool)
    for i, s in enumerate(val_samples):
        if not is_sim(s["id"]):
            continue
        key = make_key(s, mode, lastk)
        if key is None:
            continue
        hit = prior.get(key)
        if hit is None:
            continue
        logp, support = hit
        if support < min_support:
            continue
        if seen_only and support < 1:
            continue
        adj = base_logp[i] + weight * logp
        pred[i] = int(adj.argmax())
        applied[i] = True
    return pred, applied


# ----------------------------------------------------------------------------- metrics
def macro_f1(y, p, ids=None):
    if ids is not None:
        y, p = y[ids], p[ids]
    if len(y) == 0:
        return float("nan")
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)),
                    average="macro", zero_division=0)


def evaluate(y, base_pred, new_pred, sim_mask, applied):
    finding = np.isin(y, FINDING_IDS)
    changed = base_pred != new_pred
    corr = int((changed & (base_pred != y) & (new_pred == y)).sum())
    regr = int((changed & (base_pred == y) & (new_pred != y)).sum())
    au_changed = int((changed & ~sim_mask).sum())
    return {
        "full_delta": macro_f1(y, new_pred) - macro_f1(y, base_pred),
        "sim_delta": macro_f1(y, new_pred, sim_mask) - macro_f1(y, base_pred, sim_mask),
        "finding_delta": macro_f1(y, new_pred, finding) - macro_f1(y, base_pred, finding),
        "changed": int(changed.sum()),
        "applied": int(applied.sum()),
        "corrections": corr,
        "regressions": regr,
        "au_changed": au_changed,  # MUST be 0
    }


# ----------------------------------------------------------------------------- main
def main():
    samples, labels = load_samples("./data")
    yid = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = session_fold_indices(samples, labels, FOLD, N_SPLITS, SEED)
    tr = np.setdiff1d(np.arange(len(samples)), va)

    z = np.load(PROBS_NPZ, allow_pickle=True)
    assert np.array_equal(z["y_va"], yid[va]), "fold_screen y_va misaligned with regenerated fold"
    base_logp = np.log(np.clip(z[BASE_TAG].astype(np.float64), 1e-9, 1.0))

    val_samples = [samples[i] for i in va]
    y = yid[va]
    sim_mask = np.array([is_sim(s["id"]) for s in val_samples])
    base_pred = base_logp.argmax(1)
    base_f1 = macro_f1(y, base_pred)
    print(f"# E48 sequence-prior screen (leak-free fold-{FOLD} 14k val, base={BASE_TAG})")
    print(f"baseline macro-F1 = {base_f1:.4f}  |  n_val={len(y)}  sim={int(sim_mask.sum())} "
          f"au={int((~sim_mask).sum())}  finding_rows={int(np.isin(y, FINDING_IDS).sum())}\n")

    train_samples = [samples[i] for i in tr]
    train_yid = yid[tr]

    grid = []
    for mode in ("actions", "status"):
        for lastk in (0, 3, 2, 1):  # 0 = full sequence
            prior = build_prior(train_samples, train_yid, mode, lastk)
            for min_support in (1, 3, 5, 10):
                for weight in (0.3, 0.6, 1.0, 1.5):
                    new_pred, applied = apply_prior(
                        base_logp, val_samples, prior, mode, lastk,
                        weight, min_support)
                    m = evaluate(y, base_pred, new_pred, sim_mask, applied)
                    m.update(mode=mode, lastk=lastk, min_support=min_support,
                             weight=weight, n_keys=len(prior))
                    grid.append(m)

    grid.sort(key=lambda r: r["full_delta"], reverse=True)
    hdr = ("mode lastk supp w | dF1_full dF1_sim dF1_find | applied corr regr | au_chg")
    print(hdr)
    print("-" * len(hdr))
    for r in grid[:20]:
        print(f"{r['mode']:>7} {r['lastk']:>1} {r['min_support']:>2} {r['weight']:>3} | "
              f"{r['full_delta']:+.4f} {r['sim_delta']:+.4f} {r['finding_delta']:+.4f} | "
              f"{r['applied']:>5} {r['corrections']:>4} {r['regressions']:>4} | {r['au_changed']}")

    # preregistered gate (mirrors theirs): full>=+0.001 AND sim>=+0.001 AND finding>=0
    # AND corrections>regressions AND au_changed==0
    print("\n## gate check (full>=+0.001, sim>=+0.001, finding>=0, corr>regr, au_chg==0)")
    best = grid[0]
    gates = {
        "full>=+0.001": best["full_delta"] >= 0.001,
        "sim>=+0.001": best["sim_delta"] >= 0.001,
        "finding>=0": best["finding_delta"] >= 0.0,
        "corr>regr": best["corrections"] > best["regressions"],
        "au_untouched": best["au_changed"] == 0,
    }
    print(json.dumps({"best": {k: best[k] for k in
                              ("mode", "lastk", "min_support", "weight",
                               "full_delta", "sim_delta", "finding_delta",
                               "corrections", "regressions", "au_changed")},
                      "gates": gates, "PASS": all(gates.values())},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
