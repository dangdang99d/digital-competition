"""E18 LTP evaluation on the 14k val split (raw uncalibrated macro-F1).

Loads a checkpoint, optionally installs LTP + its trained thresholds, and scores the
val split in the requested mode:
  --mode off    stock model, no pruning (baseline / un-pruned F1 of a recovered ckpt)
  --mode soft   soft-mask forward (thresholds scale hidden states; training-time view)
  --mode hard   hard token pruning (drops tokens below threshold) + kept-fraction

CRITICAL split-bug guard (a wrong split silently scores on TRAIN samples): we replicate
finetune.py exactly -> split_indices is called on the label STRINGS, and ytrue is taken
from those same strings. Sanity gate: `--mode off` on e9_granite_ls must land ~0.7557.

NO logit calibration anywhere (project invariant): raw argmax, macro-F1 only.

Run: PYTHONPATH=/home/ocean/dacon python -m src.ltp_eval --ckpt <dir> --mode hard --serialize v1
"""
import argparse
import time

import numpy as np
import torch
from loguru import logger
from safetensors.torch import load_file

from src.data import (ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples,
                      macro_f1, split_indices)
from src.ltp_modeling import install_ltp, ltp_token_stats, set_ltp_mode
from src.runlog import log_cmd


def _has_ltp(ckpt, model):
    if getattr(model.config, "ltp", None):
        return True
    try:
        sd = load_file(f"{ckpt}/model.safetensors")
        return any("ltp_delta" in k for k in sd)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="ibm-granite/granite-embedding-311m-multilingual-r2",
                    help="tokenizer / base name")
    ap.add_argument("--mode", default="hard", choices=["off", "soft", "hard"])
    ap.add_argument("--serialize", default="v1")
    ap.add_argument("--final_threshold", type=float, default=0.0,
                    help="override the checkpoint's recorded final_token_threshold (0=use recorded)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    log_cmd()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    classes = list(ALL_CLASSES)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ckpt, num_labels=len(classes), torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=True,
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    ltp_on = args.mode in ("soft", "hard")
    if ltp_on:
        assert _has_ltp(args.ckpt, model), \
            f"--mode {args.mode} but checkpoint has no LTP thresholds: {args.ckpt}"
        rec = getattr(model.config, "ltp", {}) or {}
        ft = args.final_threshold or float(rec.get("final_token_threshold", 0.0))
        temp = float(rec.get("temperature", 1e-3))
        assert ft > 0, "no final_token_threshold (config.ltp missing and --final_threshold 0)"
        install_ltp(model, final_token_threshold=ft, temperature=temp, ltp_lambda=0.0,
                    mode=args.mode)
        # load the TRAINED thresholds: from_pretrained dropped the ltp_delta.* keys as
        # "unexpected" (stock model had no such params); after install_ltp they exist,
        # so re-load the full checkpoint state_dict (strict=False re-applies the
        # identical backbone/head and fills in ltp_delta).
        sd = load_file(f"{args.ckpt}/model.safetensors")
        n_loaded = sum("ltp_delta" in k for k in sd)
        assert n_loaded == len(model.model.layers), \
            f"expected {len(model.model.layers)} ltp_delta keys in ckpt, found {n_loaded}"
        model.load_state_dict(sd, strict=False)
        logger.info(f"loaded {n_loaded} trained ltp_delta params; final_threshold={ft} "
                    f"T={temp}; effective thresholds "
                    f"{[round(float(model.model.ltp_base[i]+model.model.ltp_delta[i]),5) for i in range(0, len(model.model.layers), 4)]} (every 4th layer)")
        set_ltp_mode(model, args.mode)

    # ---- correct split (STRINGS -> split, STRINGS -> ytrue) ----
    samples, labels = load_samples(args.data_dir)
    texts = build_texts(samples, input_mode="context", variant=args.serialize)
    _, va = split_indices(labels, seed=args.seed)
    ytrue = np.array([CLASS_TO_ID[labels[i]] for i in va])
    logger.info(f"val={len(va)}  mode={args.mode}  serialize={args.serialize}  ckpt={args.ckpt}")

    # length-sorted batching to cut padding waste (order restored via index map)
    order = sorted(range(len(va)), key=lambda k: len(texts[va[k]]))
    preds = np.empty(len(va), dtype=np.int64)
    layer_kept_sum = None
    n_seen = 0
    t_fwd = 0.0
    with torch.no_grad():
        for s in range(0, len(order), args.batch_size):
            pos = order[s:s + args.batch_size]
            idx = [va[k] for k in pos]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(**enc).logits
            if device == "cuda":
                torch.cuda.synchronize()
            t_fwd += time.perf_counter() - t0
            preds[pos] = logits.float().argmax(1).cpu().numpy()
            if args.mode == "hard":
                kept = ltp_token_stats(model)  # per-layer, this batch
                if kept:
                    bs = len(idx)
                    arr = np.array(kept) * bs
                    layer_kept_sum = arr if layer_kept_sum is None else layer_kept_sum + arr
                    n_seen += bs

    f1 = macro_f1(ytrue, preds)
    ms = t_fwd / len(va) * 1000.0
    logger.info("=" * 70)
    logger.info(f"RESULT  mode={args.mode}  raw macro-F1 = {f1:.4f}  "
                f"forward {ms:.2f} ms/sample (bs={args.batch_size}, eager fp32)")
    if args.mode == "hard" and layer_kept_sum is not None:
        per_layer = (layer_kept_sum / n_seen).round(3).tolist()
        mean_kept = float(np.mean(layer_kept_sum / n_seen))
        logger.info(f"        mean kept-fraction over layers = {mean_kept:.3f}")
        logger.info(f"        per-layer kept-fraction = {per_layer}")
    logger.info("=" * 70)
    print(f"\nLTP_EVAL mode={args.mode} macro_f1={f1:.4f} "
          f"kept={float(np.mean(layer_kept_sum / n_seen)) if (args.mode=='hard' and layer_kept_sum is not None) else 'NA'} "
          f"ms_per_sample={ms:.2f}")


if __name__ == "__main__":
    main()
