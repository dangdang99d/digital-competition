"""E40 char-noise — training driver. Runs the EXACT champion recipe via src.finetune.main()
and injects per-epoch free-text char noise by REPLACING src.finetune.build_dynamic_dataset
with a noising version, IN-PROCESS. Zero edits to src/. Val stays on the clean static path.

Two knobs: --p (per-char noise prob) × --alpha (fraction of samples noised each epoch, the
clean+noised MIX). The dynamic train path is triggered by passing --hist_dropout as a sentinel
(its value is IGNORED — history is NOT dropped; our dataset only char-noises free-text).

Usage:
  python -m experiments.augmentation.run_e40 --p 0.02 --alpha 0.5 [--epochs 3] [--limit N] [--tag T]
"""
import argparse
import sys

import numpy as np
import torch
from loguru import logger

from experiments.augmentation.text_noise import noise_sample_freetext

CHAMPION = [
    "--model", "ibm-granite/granite-embedding-311m-multilingual-r2",
    "--serialize", "richargs",
    "--full_data",
    "--loss", "ls", "--label_smoothing", "0.1",
    "--lr", "2e-5",
    "--grad_accum", "1",                 # granite: NO accumulation
    "--seed", "42",
    "--hist_dropout", "1.0",             # SENTINEL: routes train to the dynamic path (value ignored)
    "--out_dir", "./output/pat",
    "--results_name", "ft_results_e40.csv",
]


def make_noising_dataset(p, alpha):
    """Factory for the replacement build_dynamic_dataset (captures p, alpha)."""
    def _build(tok, samples_sub, labels, max_len, max_hist, hist_dropout, seed, variant="v1"):
        from src.data import SERIALIZE_VARIANTS, serialize
        assert variant not in ("names", "names_files"), "E40 char-noise supports serialize() variants only"
        var_kw = SERIALIZE_VARIANTS[variant]
        rng = np.random.default_rng(seed)
        logger.info(f"E40 char-noise TRAIN dataset: p={p} alpha={alpha} variant={variant} "
                    f"| free-text only, history NOT dropped (hist_dropout={hist_dropout} ignored)")

        class DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(labels)

            def __getitem__(self, i):
                s = samples_sub[i]
                if alpha > 0 and rng.random() < alpha:          # this sample noised THIS epoch
                    s = noise_sample_freetext(s, rng, p)
                text = serialize(s, max_hist=max_hist, **var_kw)
                e = tok(text, truncation=True, max_length=max_len, padding=False)
                return {"input_ids": e["input_ids"],
                        "attention_mask": e["attention_mask"],
                        "labels": int(labels[i])}

        return DS()
    return _build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, required=True, help="per-char noise probability (free-text)")
    ap.add_argument("--alpha", type=float, required=True, help="fraction of samples noised per epoch")
    ap.add_argument("--epochs", default="3")
    ap.add_argument("--batch_size", default="16", help="champion=16 (granite real bs16, no accum)")
    ap.add_argument("--max_len", default="512")
    ap.add_argument("--limit", default=None, help="subsample N train rows (smoke only)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"e40_p{args.p}_a{args.alpha}"

    # ---- inject the noising train dataset (in-process; src/ untouched) ----
    import src.finetune as ft
    ft.build_dynamic_dataset = make_noising_dataset(args.p, args.alpha)

    argv = ["src.finetune"] + CHAMPION + [
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_len", str(args.max_len),
        "--tag", tag,
    ]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    sys.argv = argv
    logger.info(f"E40 p={args.p} alpha={args.alpha} -> finetune argv: {' '.join(argv[1:])}")
    ft.main()


if __name__ == "__main__":
    main()
