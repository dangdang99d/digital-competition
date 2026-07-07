"""Prune unused embedding rows from a fine-tuned XLM-R-class checkpoint + fp16.

Why: the <=1GB submission cap. bge-m3's 250k-token embedding matrix is 45% of
its weights, but the (synthetic) competition data touches ~9.4k tokens. We keep
used-tokens ∪ the first --margin vocab ids (sentencepiece ids are frequency-
ordered, so this is a robustness margin for the hidden test) and remap at
inference: the tokenizer stays COMPLETE; script.py applies remap.npy to input
ids, sending pruned ids to <unk>. Measured unk rate on held-out data: 0.0004%
of tokens (~0.1% of samples see a single <unk>).

Usage:
  python -m src.prune_vocab --model_dir submission/model/bge-m3 \
      --out_dir submission/model/bge-m3-pruned --margin 50000
Then verify with src.verify_pruned before shipping.
"""
import argparse
import json
import os

import numpy as np
import torch
from loguru import logger

from src.data import SERIALIZE_VARIANTS, serialize


def collect_used_ids(tok, data_path, max_len, var_kw=None):
    samples = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    texts = [serialize(s, **(var_kw or {})) for s in samples]
    used = set()
    for i in range(0, len(texts), 512):
        for ids in tok(texts[i : i + 512], truncation=True, max_length=max_len).input_ids:
            used.update(ids)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_jsonl", default="./data/train.jsonl")
    ap.add_argument("--margin", type=int, default=50000,
                    help="also keep vocab ids [0, margin) as a frequency-ordered safety net")
    ap.add_argument("--max_len", type=int, default=1024,
                    help="tokenization cap when collecting used ids (use >= training max_len)")
    ap.add_argument("--serialize", default="v1", choices=sorted(SERIALIZE_VARIANTS),
                    help="serialization variant the model was trained on — MUST match so "
                         "variant-specific tokens are kept, not pruned to <unk>")
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token          # decoder-style tokenizers (Qwen3)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32)

    used = collect_used_ids(tok, args.train_jsonl, args.max_len,
                            SERIALIZE_VARIANTS[args.serialize])
    special = set(tok.all_special_ids)
    keep = sorted(used | special | set(range(args.margin)))
    logger.info(f"serialize={args.serialize}  used={len(used):,}  "
                f"keep(with margin {args.margin})={len(keep):,} of {model.config.vocab_size:,}")

    # remap: old id -> new row; pruned ids -> new position of <unk>
    old_emb = model.get_input_embeddings()
    remap = np.full(old_emb.num_embeddings, -1, dtype=np.int64)
    for new, old in enumerate(keep):
        remap[old] = new
    # byte-level BPE (Qwen3) has no <unk>: unseen text falls back to byte tokens
    # (low ids, always kept), so the remap default almost never fires — route the
    # rare pruned-but-mergeable id to eos instead.
    fallback_id = tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id
    unk_new = int(remap[fallback_id])
    assert unk_new >= 0, "fallback token must be in the keep set"
    remap[remap == -1] = unk_new

    new_emb = torch.nn.Embedding(len(keep), old_emb.embedding_dim)
    new_emb.weight.data = old_emb.weight.data[keep].clone()
    model.set_input_embeddings(new_emb)
    model.config.vocab_size = len(keep)
    model.config.pad_token_id = int(remap[tok.pad_token_id])

    model.half()
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)          # tokenizer stays complete
    np.save(os.path.join(args.out_dir, "remap.npy"), remap.astype(np.int32))

    total = sum(os.path.getsize(os.path.join(args.out_dir, f)) for f in os.listdir(args.out_dir))
    logger.success(f"pruned fp16 model -> {args.out_dir}  ({total/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
