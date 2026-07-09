"""E18 LTP parity gate (granite / ModernBERT).

Loads granite the SAME way finetune.py does (AutoModelForSequenceClassification,
attn_implementation='eager', num_labels=14, fp32), builds a batch of REAL serialized
val inputs (richargs) via src.data, then runs:
  (a) the plain stock model, and
  (b) the LTP wrapper with the soft mask forced to 1 (thresholds = -inf),
and asserts the logits are numerically identical (max abs diff < 1e-4). This proves
the wrapper is a no-op when off -- the correctness foundation before any training.

Also reports: eager forward ms/sample (plain vs soft vs hard), that attention probs
are actually materialized under eager, and the per-layer kept-token fraction hard-mode
LTP would achieve at a sample final_token_threshold.

Run:  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python -m src.ltp_parity
"""
import argparse
import time

import numpy as np
import torch
from loguru import logger

from src.data import CLASS_TO_ID, build_texts, load_samples, split_indices
from src.ltp_modeling import (install_ltp, ltp_token_stats, reset_ltp_thresholds,
                              set_ltp_mode, set_ltp_disabled_softmask)
from src.runlog import log_cmd

MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"


def load_model(device):
    from transformers import AutoModelForSequenceClassification
    classes = list(CLASS_TO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(classes), torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=True,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)},
        ignore_mismatched_sizes=True,
    )
    return model.to(device).eval()


@torch.no_grad()
def timed_forward(model, batch, iters=20):
    """Return ms/sample averaged over `iters` forward passes (excludes 3 warmups)."""
    n = batch["input_ids"].shape[0]
    for _ in range(3):
        model(**batch)
    if batch["input_ids"].is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(**batch)
    if batch["input_ids"].is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--serialize", default="richargs")
    ap.add_argument("--final_threshold", type=float, default=0.01,
                    help="sample final_token_threshold for the hard-mode telemetry")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    log_cmd()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  model={MODEL}")

    # ---- real serialized val batch (same pipeline as finetune.py) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    samples, y = load_samples(args.data_dir)
    texts = build_texts(samples, input_mode="context", variant=args.serialize)
    _, va = split_indices(y, seed=42)
    # take the longest val texts in the batch so there is real length to prune
    va_sorted = sorted(va.tolist(), key=lambda i: -len(texts[i]))
    pick = va_sorted[: args.batch_size]
    enc = tok([texts[i] for i in pick], truncation=True, max_length=args.max_len,
              padding=True, return_tensors="pt")
    batch = {"input_ids": enc["input_ids"].to(device),
             "attention_mask": enc["attention_mask"].to(device)}
    lens = enc["attention_mask"].sum(1)
    logger.info(f"batch: {args.batch_size} val samples, seq_len={batch['input_ids'].shape[1]}, "
                f"token lengths min/mean/max = {int(lens.min())}/{float(lens.float().mean()):.0f}/{int(lens.max())}")

    # ---- (a) plain stock model ----
    model = load_model(device)
    with torch.no_grad():
        plain_logits = model(**batch).logits.float().cpu()

    # confirm attentions materialize under eager
    with torch.no_grad():
        att_out = model(**batch, output_attentions=True)
    n_att = 0 if att_out.attentions is None else len(att_out.attentions)
    att_shape = tuple(att_out.attentions[0].shape) if n_att else None
    logger.info(f"eager attentions materialized: {n_att} layers, per-layer shape {att_shape}")
    assert n_att == model.config.num_hidden_layers and att_shape is not None, \
        "eager did NOT return attention probabilities — LTP scoring impossible"

    # ---- (b) LTP wrapper, soft mask forced == 1 (thresholds = -inf) ----
    install_ltp(model, final_token_threshold=args.final_threshold,
                temperature=1e-3, ltp_lambda=0.0, mode="soft")
    set_ltp_disabled_softmask(model)   # every effective threshold -> -inf  => mask==1
    with torch.no_grad():
        ltp_logits = model(**batch).logits.float().cpu()

    max_abs = (plain_logits - ltp_logits).abs().max().item()
    passed = max_abs < args.tol
    logger.info("=" * 68)
    logger.info(f"PARITY  max|Δlogit| = {max_abs:.3e}   tol = {args.tol:.0e}   "
                f"=> {'PASS' if passed else 'FAIL'}")
    logger.info("=" * 68)

    # ---- hard-mode telemetry: how many tokens LTP would keep ----
    # (thresholds were nuked to -inf for the parity test; restore the real ramp first)
    frac_nonpad = float(batch["attention_mask"].float().mean())
    logger.info(f"non-pad token fraction in batch = {frac_nonpad:.3f} (upper bound on kept)")
    set_ltp_mode(model, "hard")
    for ft in (0.005, 0.01, 0.02):
        reset_ltp_thresholds(model, final_token_threshold=ft)
        with torch.no_grad():
            model(**batch)
        kept = ltp_token_stats(model)
        logger.info(f"hard-mode kept-fraction @ final_threshold={ft}: "
                    f"per-layer {[round(k, 3) for k in kept]}  mean={np.mean(kept):.3f}")

    # ---- timing: plain vs soft(no-op) vs hard ----
    plain = load_model(device)
    ms_plain = timed_forward(plain, batch)
    set_ltp_mode(model, "soft"); set_ltp_disabled_softmask(model)
    ms_soft = timed_forward(model, batch)
    set_ltp_mode(model, "hard")
    ms_hard = timed_forward(model, batch)
    logger.info(f"forward ms/sample (eager, fp32, bs={args.batch_size}, "
                f"seq={batch['input_ids'].shape[1]}):")
    logger.info(f"  plain           = {ms_plain:.2f}")
    logger.info(f"  LTP soft (mask1)= {ms_soft:.2f}  (scoring overhead)")
    logger.info(f"  LTP hard (mask+)= {ms_hard:.2f}  (mask-extension; no unpad yet)")

    print(f"\nPARITY_RESULT max_abs_logit_diff={max_abs:.3e} tol={args.tol:.0e} "
          f"pass={passed}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
