"""First-step (step==1, no history) failure drill-down, for either model.
Learnable vs intrinsic: model vs majority prior, predicted-vs-true dist (prior
collapse?), per-class first-vs-mid error, top confusions.

Usage: python -m analysis.first_step [--model hist0|qwen3]
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

MODELS = {
    "hist0": dict(ckpt="output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500",
                  bias="output/pat/ft_BAAI__bge-m3_hist0/logit_bias.json",
                  serialize="v1", max_len=1024, pad_eos=False),
    "qwen3": dict(ckpt="output/pat/ft_Qwen__Qwen3-Embedding-0.6B/checkpoint-10500",
                  bias="output/pat/ft_Qwen__Qwen3-Embedding-0.6B/logit_bias.json",
                  serialize="v1", max_len=512, pad_eos=True),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def get_preds(cfg, cache):
    if os.path.exists(cache):
        return np.load(cache)["preds"]
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding)
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    texts = build_texts(samples, input_mode="context", max_hist=None, variant=cfg["serialize"])
    tok = AutoTokenizer.from_pretrained(cfg["ckpt"], local_files_only=True)
    if cfg["pad_eos"] and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["ckpt"], torch_dtype=torch.float16).to(DEV).eval()
    model.config.pad_token_id = tok.pad_token_id
    bmap = json.load(open(cfg["bias"]))["bias"]
    bias = np.array([bmap.get(c, 0.0) for c in ALL_CLASSES])
    enc = [tok(texts[i], truncation=True, max_length=cfg["max_len"]) for i in va]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    coll = DataCollatorWithPadding(tokenizer=tok)
    out = [0] * len(enc)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        batch = {k: v.to(DEV) for k, v in coll([enc[i] for i in idx]).items()}
        lg = model(**batch).logits.float().cpu().numpy() + bias
        for j, i in enumerate(idx):
            out[i] = int(lg[j].argmax())
    preds = np.array(out)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, preds=preds)
    return preds


def dist(ids):
    c = collections.Counter(ALL_CLASSES[i] for i in ids)
    n = sum(c.values())
    return {k: f"{v/n:.0%}" for k, v in c.most_common()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hist0", choices=list(MODELS))
    args = ap.parse_args()
    cfg = MODELS[args.model]
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])
    preds = get_preds(cfg, f"analysis/cache/{args.model}_val_preds.npz")
    step = np.array([int(samples[i]["id"].split("step_")[1]) for i in va])

    first, mid = step == 1, step >= 4
    print(f"=== MODEL: {args.model} ===")
    print(f"FIRST (step==1): n={first.sum()} acc={(preds[first]==y_true[first]).mean():.4f} "
          f"macroF1={f1_score(y_true[first], preds[first], average='macro'):.4f}")
    print(f"MID  (step>=4):  n={mid.sum()} acc={(preds[mid]==y_true[mid]).mean():.4f} "
          f"macroF1={f1_score(y_true[mid], preds[mid], average='macro'):.4f}")

    prior = collections.Counter(y_true[first]).most_common(1)[0][0]
    print(f"\nprior={ALL_CLASSES[prior]} ({(y_true[first]==prior).mean():.1%})  "
          f"model beats prior by {(preds[first]==y_true[first]).mean()-(y_true[first]==prior).mean():+.1%}")
    print("first TRUE dist:", dist(y_true[first]))
    print("first PRED dist:", dist(preds[first]), "  <- prior collapse if list_directory/plan_task inflated")

    print(f"\n{'class':20} {'first_err':>9} {'mid_err':>8} {'n':>6}")
    for c in range(len(ALL_CLASSES)):
        fm, mm = first & (y_true == c), mid & (y_true == c)
        if fm.sum() < 15:
            continue
        fe = 1 - (preds[fm] == y_true[fm]).mean()
        me = 1 - (preds[mm] == y_true[mm]).mean() if mm.sum() else float("nan")
        print(f"{ALL_CLASSES[c]:20} {fe:>8.1%} {me:>7.1%} {fm.sum():>6d}")

    conf = collections.Counter()
    for t, p in zip(y_true[first], preds[first]):
        if t != p:
            conf[(ALL_CLASSES[t], ALL_CLASSES[p])] += 1
    print("\ntop first-step confusions (true -> pred):")
    for (t, p), n in conf.most_common(8):
        print(f"  {t:18} -> {p:18} {n:4d}")


if __name__ == "__main__":
    main()
