"""Session-level error analysis: is there per-session error clustering BEYOND the
step-difficulty effect? That's the signature of session/agent-specific conventions.

Caches hist0 val predictions to analysis/cache/ so we can iterate without re-eval.
"""
import collections
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

CKPT = "output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500"
BIAS = "output/pat/ft_BAAI__bge-m3_hist0/logit_bias.json"
CACHE = "analysis/cache/hist0_val_preds.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(texts):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding)
    tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, torch_dtype=torch.float16).to(DEV).eval()
    bmap = json.load(open(BIAS))["bias"]
    bias = np.array([bmap.get(c, 0.0) for c in ALL_CLASSES])
    enc = [tok(t, truncation=True, max_length=1024) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    coll = DataCollatorWithPadding(tokenizer=tok)
    out = [0] * len(enc)
    for s in range(0, len(order), 48):
        idx = order[s:s + 48]
        batch = {k: v.to(DEV) for k, v in coll([enc[i] for i in idx]).items()}
        lg = model(**batch).logits.float().cpu().numpy() + bias
        for j, i in enumerate(idx):
            out[i] = int(lg[j].argmax())
    return np.array(out)


def main():
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])

    if os.path.exists(CACHE):
        preds = np.load(CACHE)["preds"]
        print(f"loaded cached preds ({CACHE})")
    else:
        texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
        preds = predict([texts[i] for i in va])
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, preds=preds)
    correct = (preds == y_true).astype(int)
    err = 1 - correct
    print(f"val n={len(va)}  acc={correct.mean():.4f}")

    S = [samples[i] for i in va]
    sess = np.array([s["id"].rsplit("-step_", 1)[0] for s in S])
    step = np.array([int(s["id"].split("step_")[1]) for s in S])
    gen = np.array([s["id"].split("_")[1] for s in S])

    # ---- step-difficulty base rate p(err | step) : the confound we must control ----
    step_rate = {}
    for st in np.unique(step):
        step_rate[st] = err[step == st].mean()
    exp_err = np.array([step_rate[st] for st in step])   # expected err per sample from step alone

    # ---- overdispersion test: session error variance vs binomial(given step) ----
    by_sess = collections.defaultdict(list)
    for i in range(len(va)):
        by_sess[sess[i]].append(i)
    multi = {s: ix for s, ix in by_sess.items() if len(ix) >= 3}   # need >=3 val steps
    print(f"\nsessions total={len(by_sess)}  with>=3 val steps={len(multi)}")

    # Pearson chi-square overdispersion controlling for step:
    #   under H0 (no session effect) observed errors per session ~ Poisson-binomial(exp_err)
    chi2, dof = 0.0, 0
    obs_rates, exp_rates, ns = [], [], []
    for s, ix in multi.items():
        ix = np.array(ix)
        o = err[ix].sum()
        e = exp_err[ix].sum()
        v = (exp_err[ix] * (1 - exp_err[ix])).sum()
        if v > 0:
            chi2 += (o - e) ** 2 / v
            dof += 1
        obs_rates.append(err[ix].mean()); exp_rates.append(exp_err[ix].mean()); ns.append(len(ix))
    print(f"overdispersion (session vs step-null): chi2/dof = {chi2/dof:.3f} "
          f"(=1 means NO session effect beyond step; >1.2 = real clustering)")

    # ---- intra-session error correlation: are a session's steps correct together? ----
    # compare fraction of all-correct + all-wrong sessions to the step-null expectation
    obs_allwrong = sum(1 for s, ix in multi.items() if err[np.array(ix)].mean() == 1)
    obs_allright = sum(1 for s, ix in multi.items() if err[np.array(ix)].mean() == 0)
    exp_allwrong = sum(np.prod(exp_err[np.array(ix)]) for ix in multi.values())
    exp_allright = sum(np.prod(1 - exp_err[np.array(ix)]) for ix in multi.values())
    print(f"all-WRONG sessions: obs={obs_allwrong}  vs step-null exp={exp_allwrong:.0f}")
    print(f"all-RIGHT sessions: obs={obs_allright}  vs step-null exp={exp_allright:.0f}")

    # ---- per-session error-rate distribution ----
    obs_rates = np.array(obs_rates)
    print("\nper-session error-rate distribution (>=3 val steps):")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hi = lo + 0.2 + (0.0001 if lo == 0.8 else 0)
        m = (obs_rates >= lo) & (obs_rates < hi + (1 if lo == 0.8 else 0))
        print(f"  err {lo:.1f}-{hi:.1f}: {m.sum():4d} sessions ({m.mean():5.1%})")

    # ---- characterize HARD sessions (residual = obs-exp, top decile) ----
    resid = np.array(obs_rates) - np.array(exp_rates)
    sess_keys = list(multi.keys())
    hardorder = np.argsort(-resid)
    hard = set(sess_keys[i] for i in hardorder[:max(1, len(sess_keys) // 10)])
    hardmask = np.array([s in hard for s in sess])
    print(f"\nHARD sessions (top-decile residual, n_samples={hardmask.sum()}):")
    print("  generator:", {g: f"{(gen[hardmask]==g).mean():.0%}" for g in np.unique(gen)},
          " | overall au-share:", f"{(gen=='au').mean():.0%}")
    tl = collections.Counter(ALL_CLASSES[t] for t in y_true[hardmask])
    print("  top true labels:", dict(tl.most_common(6)))
    print(f"  avg step: {step[hardmask].mean():.2f} vs overall {step.mean():.2f}")


if __name__ == "__main__":
    main()
