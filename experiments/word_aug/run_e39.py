"""E39 training driver — union pre-generated augmented rows into the t070 (AWP-Optuna best)
recipe on session-fold-0, and compare against the A0 control = t070 fold-F1 0.7806 (reused,
NOT retrained — user call). Trains FROM SCRATCH (no warm-start; E24 recovery trap).

Injection IN-PROCESS (src/ untouched): load_samples is extended with the augmented samples;
session_fold_indices splits the ORIGINALS into the fold-0 train/holdout, then appends every
augmented index to `tr` — so aug trains but NEVER enters the holdout.

Recipe = E38 t070 (fold-F1 0.7806): richargs + AWP-tuned + session-fold-0, epochs 4, best-epoch.

Usage:
  python -m experiments.word_aug.run_e39 --aug aug_cmlm_bal.jsonl --tag e39_cmlm_bal
"""
import argparse
import json
import sys

import numpy as np
from loguru import logger

# E38 t070 — best AWP-Optuna trial (fold-0 F1 0.7806); the E39 baseline recipe.
T070 = [
    "--model", "ibm-granite/granite-embedding-311m-multilingual-r2",
    "--serialize", "richargs",
    "--loss", "ls", "--label_smoothing", "0.0518",
    "--lr", "2.932e-05",
    "--batch_size", "16", "--grad_accum", "1",        # eff_batch 16, NO accumulation (granite)
    "--warmup_ratio", "0.1462",
    "--weight_decay", "3.20e-03",
    "--awp_gamma", "3.421e-03", "--awp_lr", "2.063e-04", "--awp_start_epoch", "1",
    "--group_by_length",
    "--session_fold", "0", "--session_splits", "5",
    "--max_len", "512", "--seed", "42", "--init_seed", "42",
    "--out_dir", "./output/pat", "--results_name", "ft_results_e39.csv",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", required=True, help="comma-separated augmented JSONL path(s)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", default="4")
    ap.add_argument("--batch_size", default="16")
    ap.add_argument("--max_len", default="512")
    args = ap.parse_args()

    import src.finetune as ft
    from src.data import load_samples as _orig_load
    from src.data import session_fold_indices as _orig_fold

    orig_samples, orig_y = _orig_load("./data")
    n_orig = len(orig_samples)

    aug_samples, aug_y = [], []
    for path in args.aug.split(","):
        with open(path.strip(), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                aug_y.append(r.pop("label"))
                aug_samples.append(r)
    m = len(aug_samples)
    logger.info(f"E39: {n_orig} original + {m} augmented rows from {args.aug}")

    ext_samples = orig_samples + aug_samples
    ext_y = list(orig_y) + list(aug_y)

    def _load(_data_dir="./data"):
        return ext_samples, ext_y

    def _fold(_samples, _y, fold, n_splits=5, seed=42):
        tr, va = _orig_fold(orig_samples, orig_y, fold, n_splits=n_splits, seed=seed)
        tr = np.concatenate([np.asarray(tr), np.arange(n_orig, n_orig + m)])   # aug -> train
        return tr, np.asarray(va)

    ft.load_samples = _load
    ft.session_fold_indices = _fold

    argv = ["src.finetune"] + T070 + [
        "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
        "--max_len", str(args.max_len), "--tag", args.tag,
    ]
    sys.argv = argv
    logger.info(f"E39 tag={args.tag} -> finetune argv: {' '.join(argv[1:])}")
    ft.main()


if __name__ == "__main__":
    main()
