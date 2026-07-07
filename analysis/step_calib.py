"""Step-aware logit-bias calibration vs global. Honest 2-fold: fit bias on one
half of val, measure macro-F1 on the other. Step-aware = separate bias for
first-step (step==1) vs rest, to fix the first-step prior-collapse.
"""
import json
import os

import numpy as np
import torch

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.finetune import _macro_f1, calibrate_logit_bias

CKPT = "output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500"
LCACHE = "analysis/cache/hist0_val_logits.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def logits_for(texts):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding)
    tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, torch_dtype=torch.float16).to(DEV).eval()
    enc = [tok(t, truncation=True, max_length=1024) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    coll = DataCollatorWithPadding(tokenizer=tok)
    out = np.zeros((len(enc), len(ALL_CLASSES)), dtype=np.float32)
    for s in range(0, len(order), 48):
        idx = order[s:s + 48]
        batch = {k: v.to(DEV) for k, v in coll([enc[i] for i in idx]).items()}
        lg = model(**batch).logits.float().cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = lg[j]
    return out


def main():
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    labels = np.array([CLASS_TO_ID[y[i]] for i in va])
    step = np.array([int(samples[i]["id"].split("step_")[1]) for i in va])

    if os.path.exists(LCACHE):
        logits = np.load(LCACHE)["logits"]
    else:
        texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
        logits = logits_for([texts[i] for i in va])
        os.makedirs(os.path.dirname(LCACHE), exist_ok=True)
        np.savez(LCACHE, logits=logits)

    first = step == 1
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 2, size=len(va))       # random 2-fold

    res = {k: [] for k in ("global_all", "step_all", "global_first", "step_first",
                           "global_mid", "step_mid")}
    for te in (0, 1):
        tr_m, te_m = fold != te, fold == te
        # --- global bias fit on train half ---
        bg, _, _ = calibrate_logit_bias(logits[tr_m], labels[tr_m])
        # --- step-aware: separate bias for first vs rest ---
        bf, _, _ = calibrate_logit_bias(logits[tr_m & first], labels[tr_m & first])
        bm, _, _ = calibrate_logit_bias(logits[tr_m & ~first], labels[tr_m & ~first])

        def evalf(mask, bias):
            return _macro_f1(logits[mask], labels[mask], bias)

        # step-aware predictions on eval half: apply bf to first, bm to rest
        def step_macro(mask):
            m1, m0 = mask & first, mask & ~first
            preds = np.empty(mask.sum(), dtype=int)
            sub = np.where(mask)[0]
            bmap = {i: (bf if first[i] else bm) for i in sub}
            from sklearn.metrics import f1_score
            pr = np.array([np.argmax(logits[i] + bmap[i]) for i in sub])
            return f1_score(labels[mask], pr, labels=list(range(14)),
                            average="macro", zero_division=0)

        res["global_all"].append(evalf(te_m, bg))
        res["step_all"].append(step_macro(te_m))
        res["global_first"].append(evalf(te_m & first, bg))
        res["step_first"].append(evalf(te_m & first, bf))
        res["global_mid"].append(evalf(te_m & ~first, bg))
        res["step_mid"].append(evalf(te_m & ~first, bm))

    print(f"first-step samples: {first.sum()} / {len(va)}  ({first.mean():.0%})")
    print(f"\n{'metric':16} {'global':>8} {'step-aware':>11} {'Δ':>8}")
    for scope, g, s in (("overall", "global_all", "step_all"),
                        ("first-step", "global_first", "step_first"),
                        ("mid/rest", "global_mid", "step_mid")):
        gm, sm = np.mean(res[g]), np.mean(res[s])
        print(f"{scope:16} {gm:>8.4f} {sm:>11.4f} {sm-gm:>+8.4f}")
    print("\n(held-out 2-fold macro-F1; global bias fit on all steps, step-aware fits first/rest separately)")


if __name__ == "__main__":
    main()
