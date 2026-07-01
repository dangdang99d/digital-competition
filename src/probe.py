"""Frozen-backbone linear probe over pretrained encoders — model-selection sweep.

Goal: identify which pretrained encoder produces the best frozen representation for the
14-class next-action task, measured by the competition metric (Macro-F1).

For each model in models.yaml:
  1. embed all inputs with the FROZEN backbone (no gradients), cache to disk,
  2. train a linear classifier (multinomial logistic regression) on the train split,
  3. report Macro-F1 on the held-out val split.

Frozen + linear = a true linear probe: it measures representation quality, not fine-tuning.

Usage:
  python -m src.probe --models src/models.yaml --input context
  python -m src.probe --only Qwen/Qwen3-Embedding-0.6B      # single model
Outputs: output/probe_results.csv  and  cached embeddings in output/emb_cache/.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import yaml
from loguru import logger
from sklearn.linear_model import LogisticRegression

from src.data import (
    ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, macro_f1, split_indices,
)


# ----------------------------- embedding -----------------------------

def mean_pool(last_hidden, attn_mask):
    m = attn_mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def cls_pool(last_hidden, attn_mask):
    # first token ([CLS]); models with right padding put it at index 0
    return last_hidden[:, 0]


def last_token_pool(last_hidden, attn_mask):
    # left-padded -> index -1; else gather at each row's true length-1
    left_padded = (attn_mask[:, -1].sum() == attn_mask.shape[0])
    if left_padded:
        return last_hidden[:, -1]
    lengths = attn_mask.sum(1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]


POOLERS = {"mean": mean_pool, "last_token": last_token_pool, "cls": cls_pool}


@torch.no_grad()
def embed_texts(texts, cfg, device, batch_size=64):
    """Embed all texts with a frozen encoder. Returns (N, D) float32 L2-normalized array."""
    from transformers import AutoModel, AutoTokenizer

    mid = cfg["id"]
    pool_fn = POOLERS[cfg.get("pool", "mean")]
    prefix = cfg.get("prefix", "")
    max_len = int(cfg.get("max_len", 512))
    trc = bool(cfg.get("trust_remote_code", False))
    pad_side = "left" if cfg.get("pool") == "last_token" else "right"

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=trc, padding_side=pad_side)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(mid, trust_remote_code=trc, torch_dtype=dtype)
    model = model.to(device).eval()

    out = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        emb = pool_fn(h, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.float().cpu().numpy())
        if (i // batch_size) % 20 == 0:
            logger.info(f"  [{mid}] embedded {i + len(batch)}/{len(texts)}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(out)


def cache_key(cfg, input_mode, n):
    """Stable filename for cached embeddings of (model, input_mode, dataset size)."""
    raw = f"{cfg['id']}|{cfg.get('pool')}|{cfg.get('prefix')}|{cfg.get('max_len')}|{input_mode}|{n}"
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    safe = cfg["id"].replace("/", "__")
    return f"{safe}__{input_mode}__{h}.npy"


def get_embeddings(texts, cfg, input_mode, device, cache_dir, batch_size):
    """Load embeddings from cache or compute + save them."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_key(cfg, input_mode, len(texts)))
    if os.path.exists(path):
        logger.info(f"  [{cfg['id']}] cache hit -> {path}")
        return np.load(path)
    t0 = time.time()
    emb = embed_texts(texts, cfg, device, batch_size=batch_size)
    np.save(path, emb)
    logger.info(f"  [{cfg['id']}] embedded {emb.shape} in {time.time() - t0:.1f}s -> cached")
    return emb


# ----------------------------- probe -----------------------------

def run_probe(emb, y_ids, tr, va, C=2.0, seed=42):
    """Train multinomial LogReg on frozen embeddings; return (val_macro_f1, train_macro_f1)."""
    # sklearn >=1.7 is multinomial by default for multiclass; no multi_class arg.
    clf = LogisticRegression(
        max_iter=2000, C=C, class_weight="balanced", random_state=seed,
    )
    clf.fit(emb[tr], y_ids[tr])
    va_pred = clf.predict(emb[va])
    tr_pred = clf.predict(emb[tr])
    return macro_f1(y_ids[va], va_pred), macro_f1(y_ids[tr], tr_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--models", default="src/models.yaml")
    ap.add_argument("--only", default=None, help="run just this model id")
    ap.add_argument("--input", default="context", choices=["context", "prompt"])
    ap.add_argument("--max_hist", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./output")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  input={args.input}")

    # data
    samples, y = load_samples(args.data_dir)
    texts = build_texts(samples, input_mode=args.input, max_hist=args.max_hist)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=args.seed)
    logger.info(f"samples={len(texts)}  train={len(tr)}  val={len(va)}  classes={len(ALL_CLASSES)}")

    with open(args.models) as f:
        model_cfgs = yaml.safe_load(f)["models"]
    if args.only:
        model_cfgs = [c for c in model_cfgs if c["id"] == args.only]
        if not model_cfgs:
            raise SystemExit(f"--only {args.only} not in {args.models}")

    cache_dir = os.path.join(args.out_dir, "emb_cache")
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for cfg in model_cfgs:
        mid = cfg["id"]
        logger.info(f"=== {mid} ===")
        try:
            emb = get_embeddings(texts, cfg, args.input, device, cache_dir, args.batch_size)
            t0 = time.time()
            val_f1, train_f1 = run_probe(emb, y_ids, tr, va, C=args.C, seed=args.seed)
            row = {
                "model": mid, "input": args.input, "dim": emb.shape[1],
                "val_macro_f1": round(val_f1, 4), "train_macro_f1": round(train_f1, 4),
                "probe_sec": round(time.time() - t0, 1),
            }
            results.append(row)
            logger.success(f"  {mid}: val Macro-F1={val_f1:.4f} (train {train_f1:.4f})")
        except Exception as e:  # keep sweeping even if one model fails
            logger.error(f"  {mid} FAILED: {type(e).__name__}: {e}")
            results.append({"model": mid, "input": args.input, "error": str(e)})

    # write results
    import csv as _csv
    out_csv = os.path.join(args.out_dir, "probe_results.csv")
    keys = ["model", "input", "dim", "val_macro_f1", "train_macro_f1", "probe_sec", "error"]
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})
    logger.info(f"wrote {out_csv}")

    # leaderboard to stdout
    ok = [r for r in results if "val_macro_f1" in r]
    ok.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    print("\n=== Linear-probe leaderboard (val Macro-F1) ===")
    for r in ok:
        print(f"  {r['val_macro_f1']:.4f}  {r['model']}  (dim={r['dim']})")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
