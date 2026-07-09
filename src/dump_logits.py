"""Dump a fine-tuned checkpoint's logits over ALL train samples, in load_samples
order, for distillation (--distill_from in src.finetune).

fp16 inference (verified: 100% prediction agreement with fp32 on this task) +
length-sorted batches, same tricks as the submission script.

Usage:
  python -m src.dump_logits --ckpt output/pat/ft_Qwen__Qwen3-Embedding-0.6B/checkpoint-10500 \
      --out output/pat/teacher_logits_qwen3.npz
"""
import argparse

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from src.data import build_texts, load_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=512,
                    help="use the length the teacher was TRAINED at")
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--out", required=True)
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

    samples, _ = load_samples(args.data_dir)
    texts = build_texts(samples)  # full history — matches how the teachers trained
    order = np.argsort([len(t) for t in texts])  # length-sorted: ~2x fewer pad FLOPs
    logits = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(0, len(order), args.batch_size), mininterval=5.0):
            idx = order[i:i + args.batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to("cuda")
            logits[idx] = model(**enc).logits.float().cpu().numpy()

    np.savez(args.out, logits=logits)
    logger.success(f"{logits.shape} teacher logits -> {args.out}")


if __name__ == "__main__":
    main()
