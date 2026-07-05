"""Per-sample debug dump: run a checkpoint on the val split and write ONE JSON
per sample containing everything needed to debug that prediction in one place:

  {
    "id", "correct", "true", "pred", "pred_calibrated",
    "logits":            {action: value, ...}   raw model outputs
    "logits_calibrated": {action: value, ...}   after logit_bias.json
    "segment_shares":    {"meta": .., "history": .., "prompt": ..}
    "tokens":            [[token_text, saliency], ...]   gradient-x-input per token
    "text":              the whole serialized input string
  }

Saliency (the expensive gradient pass) is computed for wrong samples by default
(--saliency wrong|all|none); correct samples still get logits/predictions.
An index file _summary.csv is written alongside for quick filtering.

Usage:
  python -m src.debug_dump --ckpt submission/model/bge-m3-fp32-backup \
      --out_dir output/val_debug_bgem3
"""
import argparse
import csv as _csv
import json
import os

import numpy as np
import torch
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, serialize, split_indices
from src.saliency import segment_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=512,
                    help="MUST match the checkpoint's training max_len")
    ap.add_argument("--saliency", choices=["wrong", "all", "none"], default="wrong")
    ap.add_argument("--batch_size", type=int, default=4, help="for the gradient pass")
    ap.add_argument("--limit", type=int, default=0, help="cap val samples (0=all); for tests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./output/val_debug")
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ckpt, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ckpt, local_files_only=True, torch_dtype=torch.float32).to(device).eval()
    coll = DataCollatorWithPadding(tokenizer=tok)

    bias_path = os.path.join(args.ckpt, "logit_bias.json")
    bias = np.zeros(len(ALL_CLASSES))
    if os.path.exists(bias_path):
        bmap = json.load(open(bias_path))["bias"]
        bias = np.array([bmap.get(c, 0.0) for c in ALL_CLASSES])

    samples, y = load_samples(args.data_dir)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    _, va = split_indices(y, seed=args.seed)
    if args.limit:
        va = va[: args.limit]
    texts = {i: serialize(samples[i]) for i in va}

    # ---- pass 1: logits for every val sample (no grad, length-sorted) ----
    # tokenize once WITHOUT truncation to know each sample's true length, then
    # cap to max_len for the model (mirrors training/inference behavior)
    full_len = {}
    encs = {}
    for i in va:
        e = tok(texts[i], truncation=False, padding=False)
        full_len[i] = len(e["input_ids"])
        encs[i] = {"input_ids": e["input_ids"][: args.max_len],
                   "attention_mask": e["attention_mask"][: args.max_len]}
    order = sorted(va, key=lambda i: len(encs[i]["input_ids"]), reverse=True)
    logits_all = {}
    with torch.no_grad():
        for s in range(0, len(order), 96):
            chunk = order[s:s + 96]
            b = {k: v.to(device) for k, v in coll([encs[i] for i in chunk]).items()}
            out = model(**b).logits.float().cpu().numpy()
            for j, i in enumerate(chunk):
                logits_all[i] = out[j]
    preds = {i: int(logits_all[i].argmax()) for i in va}
    preds_cal = {i: int((logits_all[i] + bias).argmax()) for i in va}
    n_wrong = sum(preds[i] != y_ids[i] for i in va)
    logger.info(f"val={len(va)}  wrong={n_wrong}  acc={1 - n_wrong/len(va):.4f}")

    # ---- pass 2: saliency (gradient-x-input) for the selected set ----
    if args.saliency == "all":
        sal_set = list(va)
    elif args.saliency == "wrong":
        sal_set = [i for i in va if preds[i] != y_ids[i]]
    else:
        sal_set = []
    logger.info(f"saliency pass on {len(sal_set)} samples")

    emb_layer = model.get_input_embeddings()
    model.gradient_checkpointing_enable()   # backward on 8GB GPUs (trades compute for VRAM)
    saliency = {}          # i -> (tokens list, shares dict)
    sal_sorted = sorted(sal_set, key=lambda i: len(encs[i]["input_ids"]), reverse=True)
    for s in range(0, len(sal_sorted), args.batch_size):
        chunk = sal_sorted[s:s + args.batch_size]
        enc = tok([texts[i] for i in chunk], truncation=True, max_length=args.max_len,
                  padding=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        embeds = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
        out = model(inputs_embeds=embeds, attention_mask=enc["attention_mask"]).logits
        out.gather(1, out.argmax(-1, keepdim=True)).sum().backward()
        sal = ((embeds.grad * embeds).norm(dim=-1) * enc["attention_mask"]).detach().cpu().numpy()
        for j, i in enumerate(chunk):
            meta_end, prompt_start = segment_spans(texts[i])
            shares = {"meta": 0.0, "history": 0.0, "prompt": 0.0}
            toks = []
            for t, (a, b) in enumerate(offsets[j].tolist()):
                if a == b:
                    continue
                seg = "meta" if b <= meta_end else ("prompt" if a >= prompt_start else "history")
                v = float(sal[j, t])
                shares[seg] += v
                toks.append([texts[i][a:b], round(v, 5)])
            tot = sum(shares.values()) or 1.0
            saliency[i] = (toks, {k: round(v / tot, 4) for k, v in shares.items()})
        if (s // args.batch_size) % 25 == 0:
            logger.info(f"  saliency {s + len(chunk)}/{len(sal_sorted)}")

    # ---- write one JSON per sample (split by correctness) + a summary index ----
    os.makedirs(os.path.join(args.out_dir, "incorrect"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "correct"), exist_ok=True)
    rows = []
    for i in va:
        sid = samples[i]["id"]
        rec = {
            "id": sid,
            "correct": bool(preds[i] == y_ids[i]),
            "true": ALL_CLASSES[y_ids[i]],
            "pred": ALL_CLASSES[preds[i]],
            "pred_calibrated": ALL_CLASSES[preds_cal[i]],
            "logits": {c: round(float(v), 4) for c, v in zip(ALL_CLASSES, logits_all[i])},
            "logits_calibrated": {c: round(float(v), 4)
                                  for c, v in zip(ALL_CLASSES, logits_all[i] + bias)},
            # truncated=True: input exceeded max_len, so the model never saw the
            # tail (incl. the PROMPT: line) — prediction made from history alone
            "truncated": full_len[i] > args.max_len,
            "n_tokens_full": full_len[i],
            "n_tokens_used": len(encs[i]["input_ids"]),
            "text": texts[i],
        }
        if i in saliency:
            rec["tokens"], rec["segment_shares"] = saliency[i]
        sub = "correct" if rec["correct"] else "incorrect"
        with open(os.path.join(args.out_dir, sub, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        rows.append({"id": sid, "correct": rec["correct"], "true": rec["true"],
                     "pred": rec["pred"], "pred_calibrated": rec["pred_calibrated"],
                     "margin": round(float(np.sort(logits_all[i])[-1] - np.sort(logits_all[i])[-2]), 4),
                     "truncated": rec["truncated"], "has_saliency": i in saliency})
    rows.sort(key=lambda r: (r["correct"], r["margin"]))     # wrong & least-confident first
    with open(os.path.join(args.out_dir, "_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.success(f"{len(rows)} sample JSONs + _summary.csv -> {args.out_dir}")


if __name__ == "__main__":
    main()
