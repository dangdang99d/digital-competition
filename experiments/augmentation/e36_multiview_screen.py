"""E36 arm A — inference-time multi-view averaging screen (professor's literal idea).

Take ONE trained model, serialize each val row into K *views*, softmax-average the
model's outputs across the views, argmax, macro-F1. Compare vs the single training
view. Zero retraining. Honest held-out: e9_granite_ls is v1+LS trained on the seed-42
56k split, so the 14k val is genuinely held out.

View axes tested (all keep the model near its v1 training distribution):
  - history-length truncation: max_hist in {None(full),12,8,6,4}  (PRIMARY, in-dist)
  - serialization variant: v1 / nometa / leanact                  (secondary, milder)

Baseline = single view (v1, full history) — exactly how the model was trained/evaluated.
Reports every single view + a set of averaged combos so we can see if ANY averaging beats
the single training view on the honest val.
"""
import argparse
import itertools

import numpy as np
import torch
from loguru import logger

from src.data import (ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples,
                      macro_f1, split_indices)


def forward_logits(model, tok, texts, max_len, bs):
    order = np.argsort([len(t) for t in texts])  # length-sorted, fewer pad FLOPs
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            out[idx] = model(**enc).logits.float().cpu().numpy()
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/pat/ft_ibm-granite__granite-embedding-"
                    "311m-multilingual-r2_e9_granite_ls/checkpoint-10500")
    ap.add_argument("--variant", default="v1", help="the model's TRAINED serialization")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="experiments/augmentation/e36_multiview_probs.npz")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
    model.config.pad_token_id = tok.pad_token_id

    samples, y = load_samples(args.data_dir)
    _, va = split_indices(y, seed=42)                 # honest 14k val
    val_samples = [samples[i] for i in va]
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])

    # ---- define the views (name -> build_texts kwargs) ----
    views = {}
    # history-length truncation, model's trained variant (PRIMARY)
    for mh in [None, 12, 8, 6, 4]:
        name = f"{args.variant}_h{'full' if mh is None else mh}"
        views[name] = dict(variant=args.variant, max_hist=mh)
    # serialization-variant views at full history (secondary)
    for v in ["v1", "nometa", "leanact"]:
        if v != args.variant:
            views[f"{v}_hfull"] = dict(variant=v, max_hist=None)

    # ---- forward pass once per view, cache softmax probs ----
    probs = {}
    for name, kw in views.items():
        texts = build_texts(val_samples, **kw)
        logits = forward_logits(model, tok, texts, args.max_len, args.batch_size)
        p = softmax(logits)
        probs[name] = p
        f1 = macro_f1(y_true, p.argmax(1))
        logger.info(f"[single] {name:16s} macro-F1 = {f1:.4f}")

    baseline_name = f"{args.variant}_hfull"
    base_f1 = macro_f1(y_true, probs[baseline_name].argmax(1))
    logger.info(f"\nBASELINE single training view {baseline_name}: {base_f1:.4f}\n")

    # ---- averaged combos ----
    def avg_f1(names):
        p = np.mean([probs[n] for n in names], axis=0)
        return macro_f1(y_true, p.argmax(1))

    trunc = [f"{args.variant}_h{s}" for s in ['full', 12, 8, 6, 4]]
    combos = {
        "avg trunc[full,12,8]":      trunc[:3],
        "avg trunc[full,12,8,6]":    trunc[:4],
        "avg trunc[full,12,8,6,4]":  trunc[:5],
        "avg trunc[full,12]":        [trunc[0], trunc[1]],
        "avg trunc[full,8]":         [trunc[0], trunc[2]],
    }
    var_views = [n for n in probs if n.endswith("_hfull")]
    if len(var_views) > 1:
        combos["avg all-variant hfull"] = var_views
        combos["avg [v1full,trunc12,leanact]"] = [baseline_name, trunc[1]] + \
            [n for n in var_views if n.startswith("leanact")]

    results = {}
    for label, names in combos.items():
        f1 = avg_f1(names)
        results[label] = f1
        d = f1 - base_f1
        flag = "  <== beats baseline" if d > 0 else ""
        logger.info(f"[avg]  {label:32s} macro-F1 = {f1:.4f}  (Δ {d:+.4f}){flag}")

    np.savez(args.out, y_true=y_true, **{f"probs__{k}": v for k, v in probs.items()})
    logger.success(f"probs cached -> {args.out}")
    logger.info(f"\nSUMMARY: baseline {base_f1:.4f} | best avg "
                f"{max(results.values()):.4f} "
                f"(Δ {max(results.values())-base_f1:+.4f})")


if __name__ == "__main__":
    main()
