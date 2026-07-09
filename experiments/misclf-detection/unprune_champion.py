"""Reconstruct the UNPRUNED qwen3 champion from the vocab-pruned submission.

The champion weights only survive inside submit_0703_qwen3_pruned.zip, which ships the
full 151,669-piece tokenizer + a pruned 25,298-row embedding + remap.npy (full-vocab id
-> kept row). Vocab pruning is lossless on our data (kept tokens map bijectively to their
own row), so we rebuild the full embedding  E_full[t] = E_pruned[remap[t]]  and save a
plain standard checkpoint. Downstream (M0) then loads it with the normal tokenizer, no
remap. Correctness is confirmed later by M0's parity check vs analysis/cache/qwen3_val_logits.npz.

Run:
  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/misclf-detection/unprune_champion.py \
    --pruned <scratch>/model/qwen3-0.6b --out output/pat/qwen3_champion_unpruned
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned", required=True, help="extracted pruned model dir (has remap.npy)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pruned = Path(args.pruned)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    remap = np.load(pruned / "remap.npy")               # (V_full,) full id -> pruned row
    V_full = len(remap)
    model = AutoModelForSequenceClassification.from_pretrained(
        pruned, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True)

    emb = model.get_input_embeddings().weight.data       # (V_pruned, H)
    V_pruned, H = emb.shape
    logger.info(f"pruned embed {tuple(emb.shape)} -> full vocab {V_full}")
    full = emb[torch.from_numpy(remap.astype(np.int64))]  # (V_full, H) gather -> lossless on kept ids

    new = torch.nn.Embedding(V_full, H, dtype=emb.dtype)
    new.weight.data.copy_(full)
    model.set_input_embeddings(new)
    model.config.vocab_size = V_full
    # eos/pad in full-vocab space (Qwen3 = 151643); it maps to the pruned pad row
    pad_full = int(np.where(remap == model.config.pad_token_id)[0][0]) \
        if model.config.pad_token_id < V_pruned else 151643
    model.config.pad_token_id = pad_full
    logger.info(f"full-vocab pad_token_id = {pad_full}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    # ship the full tokenizer as-is (it was never pruned)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "added_tokens.json", "merges.txt", "vocab.json"):
        src = pruned / f
        if src.exists():
            shutil.copy(src, Path(args.out) / f)
    tok = AutoTokenizer.from_pretrained(args.out, trust_remote_code=True)
    logger.success(f"unpruned champion -> {args.out}  (vocab {V_full}, tokenizer {tok.vocab_size})")


if __name__ == "__main__":
    main()
