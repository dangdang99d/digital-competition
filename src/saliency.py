"""Saliency diagnostic (#5): which parts of the input does the model actually use?

For a sample of val items (all wrong predictions + an equal number of correct
ones), computes gradient-x-input attribution per token, then aggregates
|attribution| into three segments of our serialization:
  meta   — the [tier=... ci=... open=...] bracket line
  history— USER:/ACTION lines
  prompt — the final PROMPT: line
Outputs per-sample rows (saliency shares + top tokens) and logs aggregate
shares for wrong vs correct predictions. If wrong predictions systematically
under-attend the prompt (or drown in history), that motivates history capping
(see hist_ablation.sbatch) or serialization changes.

Usage:
  python -m src.saliency --ckpt output/ft_BAAI__bge-m3/checkpoint-10500 \
      --n_wrong 400 --out output/saliency_bgem3.csv
"""
import argparse
import csv as _csv

import numpy as np
import torch
from loguru import logger

from src.data import CLASS_TO_ID, ALL_CLASSES, load_samples, serialize, split_indices


def segment_spans(text):
    """Character spans of (meta, history, prompt) in the serialized text."""
    lines = text.split("\n")
    pos, spans = 0, []
    for ln in lines:
        spans.append((pos, pos + len(ln), ln))
        pos += len(ln) + 1
    meta_end = spans[0][1]
    prompt_start = spans[-1][0]
    return meta_end, prompt_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--n_wrong", type=int, default=400,
                    help="wrong-prediction samples to analyze (+ same number correct)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./output/saliency.csv")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ckpt, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ckpt, local_files_only=True, torch_dtype=torch.float32).to(device).eval()

    samples, y = load_samples(args.data_dir)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    _, va = split_indices(y, seed=args.seed)
    texts = {i: serialize(samples[i]) for i in va}

    # ---- pass 1: plain predictions to find wrong/correct sets ----
    preds = {}
    with torch.no_grad():
        idxs = list(va)
        for s in range(0, len(idxs), 64):
            chunk = idxs[s:s + 64]
            enc = tok([texts[i] for i in chunk], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            for i, p in zip(chunk, logits.argmax(-1).tolist()):
                preds[i] = p
    wrong = [i for i in va if preds[i] != y_ids[i]][: args.n_wrong]
    right = [i for i in va if preds[i] == y_ids[i]][: len(wrong)]
    logger.info(f"val={len(va)}  wrong={sum(preds[i] != y_ids[i] for i in va)}  "
                f"analyzing {len(wrong)} wrong + {len(right)} correct")

    emb_layer = model.get_input_embeddings()

    def saliency_rows(indices, group):
        rows = []
        for s in range(0, len(indices), args.batch_size):
            chunk = indices[s:s + args.batch_size]
            enc = tok([texts[i] for i in chunk], truncation=True, max_length=args.max_len,
                      padding=True, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")
            enc = {k: v.to(device) for k, v in enc.items()}
            embeds = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
            out = model(inputs_embeds=embeds, attention_mask=enc["attention_mask"]).logits
            out.gather(1, out.argmax(-1, keepdim=True)).sum().backward()
            sal = (embeds.grad * embeds).norm(dim=-1)          # (B, L) grad-x-input magnitude
            sal = (sal * enc["attention_mask"]).detach().cpu().numpy()
            for j, i in enumerate(chunk):
                meta_end, prompt_start = segment_spans(texts[i])
                shares = {"meta": 0.0, "history": 0.0, "prompt": 0.0}
                tok_sals = []
                for t, (a, b) in enumerate(offsets[j].tolist()):
                    if a == b:      # special/pad
                        continue
                    seg = "meta" if b <= meta_end else ("prompt" if a >= prompt_start else "history")
                    shares[seg] += float(sal[j, t])
                    tok_sals.append((float(sal[j, t]), texts[i][a:b]))
                tot = sum(shares.values()) or 1.0
                tok_sals.sort(reverse=True)
                rows.append({
                    "id": samples[i]["id"], "group": group,
                    "true": ALL_CLASSES[y_ids[i]], "pred": ALL_CLASSES[preds[i]],
                    "meta_share": round(shares["meta"] / tot, 4),
                    "history_share": round(shares["history"] / tot, 4),
                    "prompt_share": round(shares["prompt"] / tot, 4),
                    "top_tokens": " | ".join(t for _, t in tok_sals[:10]),
                })
        return rows

    rows = saliency_rows(wrong, "wrong") + saliency_rows(right, "correct")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"per-sample saliency -> {args.out}")

    for g in ("wrong", "correct"):
        sub = [r for r in rows if r["group"] == g]
        for k in ("meta_share", "history_share", "prompt_share"):
            logger.info(f"  {g:8s} mean {k}: {np.mean([r[k] for r in sub]):.3f}")


if __name__ == "__main__":
    main()
