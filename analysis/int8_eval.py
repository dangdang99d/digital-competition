"""int8 (bitsandbytes LLM.int8) vs fp16 inference on the val split — the packaging-
track quantization check: what does int8 cost in Macro-F1, and what does it buy in
memory? (Speed here is 4060 speed; T4 int8 is typically NOT faster than fp16 —
the win is memory/zip headroom only.)

Usage:
  python -m analysis.int8_eval --ckpt output/pat/ft_Qwen__Qwen3-Embedding-0.6B/checkpoint-10500
Prints a comparison table; logits for both precisions go to analysis/int8_logits.npz.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.finetune import calibrate_logit_bias


def run_eval(ckpt, texts, max_len, batch_size, int8):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"trust_remote_code": True}
    if int8:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, **kwargs)
    if not int8:
        model = model.cuda()
    model.eval()
    model.config.pad_token_id = tok.pad_token_id

    torch.cuda.reset_peak_memory_stats()
    order = np.argsort([len(t) for t in texts])       # length-sorted, like submission
    logits = np.zeros((len(texts), len(ALL_CLASSES)), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(order), batch_size), mininterval=5.0,
                      desc="int8" if int8 else "fp16"):
            idx = order[i:i + batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            logits[idx] = model(**enc).logits.float().cpu().numpy()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return logits, dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    samples, y = load_samples(args.data_dir)
    texts = build_texts(samples)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    _, va = split_indices(y, seed=42)
    va_texts = [texts[i] for i in va]
    va_y = y_ids[va]
    logger.info(f"val={len(va)}  ckpt={args.ckpt}")

    results = {}
    for tag, int8 in [("fp16", False), ("int8", True)]:
        logits, dt, peak = run_eval(args.ckpt, va_texts, args.max_len, args.batch_size, int8)
        bias, base_f1, tuned_f1 = calibrate_logit_bias(logits, va_y)
        results[tag] = {"logits": logits, "sec": dt, "peak_gb": peak,
                        "raw_f1": base_f1, "cal_f1": tuned_f1}
        logger.info(f"{tag}: raw {base_f1:.4f}  calibrated {tuned_f1:.4f}  "
                    f"{dt:.0f}s  peak {peak:.2f} GB")

    agree = (results["fp16"]["logits"].argmax(1) == results["int8"]["logits"].argmax(1)).mean()
    print(f"\n{'':>6} {'raw F1':>8} {'cal F1':>8} {'sec':>6} {'peak GB':>8}")
    for tag in ("fp16", "int8"):
        r = results[tag]
        print(f"{tag:>6} {r['raw_f1']:>8.4f} {r['cal_f1']:>8.4f} {r['sec']:>6.0f} {r['peak_gb']:>8.2f}")
    print(f"\nprediction agreement fp16 vs int8: {agree:.4f}")
    print(f"delta cal F1 (int8 - fp16): {results['int8']['cal_f1'] - results['fp16']['cal_f1']:+.4f}")

    os.makedirs("analysis", exist_ok=True)
    np.savez("analysis/int8_logits.npz",
             fp16=results["fp16"]["logits"], int8=results["int8"]["logits"], val_idx=va)
    with open("analysis/int8_eval.json", "w") as f:
        json.dump({t: {k: float(v) for k, v in r.items() if k != "logits"}
                   for t, r in results.items()} | {"agreement": float(agree)}, f, indent=2)
    logger.success("saved -> analysis/int8_logits.npz, analysis/int8_eval.json")


if __name__ == "__main__":
    main()
