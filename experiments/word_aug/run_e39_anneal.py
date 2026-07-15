"""E39 annealing variant — 'data annealing': first --anneal_after epochs train on
original+augmented, remaining epochs on ORIGINAL-ONLY (aug turned off), in ONE continuous run
(single LR schedule, no checkpoint restart -> avoids the E24 warm-start degradation confound).

Mechanism (keeps step count / LR schedule identical to the plain aug run): the train dataset
length is held constant, but from epoch `anneal_after` onward each augmented slot returns a
RANDOM ORIGINAL sample instead — so the latter epochs see clean data only (clean oversampled to
fill the aug slots). Everything else = run_e39 / t070 recipe.

  python -m experiments.word_aug.run_e39_anneal --aug qwen_blanket.jsonl --tag e39_qwen_bln_anneal --anneal_after 2
"""
import argparse
import json
import sys

import numpy as np
from loguru import logger

from experiments.word_aug.run_e39 import T070


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", default="4")
    ap.add_argument("--anneal_after", type=int, default=2,
                    help="epochs [0,anneal_after) use aug+clean; [anneal_after,epochs) clean-only")
    ap.add_argument("--batch_size", default="16")
    ap.add_argument("--max_len", default="512")
    args = ap.parse_args()

    import torch
    import src.finetune as ft
    from src.data import load_samples as _orig_load
    from src.data import session_fold_indices as _orig_fold

    orig_samples, orig_y = _orig_load("./data")
    n_orig = len(orig_samples)

    aug_samples, aug_y = [], []
    for path in args.aug.split(","):
        with open(path.strip(), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    aug_y.append(r.pop("label"))
                    aug_samples.append(r)
    m = len(aug_samples)
    logger.info(f"E39-anneal: {n_orig} original + {m} augmented; aug OFF from epoch {args.anneal_after}")

    ext_samples = orig_samples + aug_samples
    ext_y = list(orig_y) + list(aug_y)
    ft.load_samples = lambda _d="./data": (ext_samples, ext_y)

    def _fold(_s, _y, fold, n_splits=5, seed=42):
        tr, va = _orig_fold(orig_samples, orig_y, fold, n_splits=n_splits, seed=seed)
        tr = np.concatenate([np.asarray(tr), np.arange(n_orig, n_orig + m)])   # aug -> train tail
        return tr, np.asarray(va)
    ft.session_fold_indices = _fold

    # ---- annealing injection (length-constant; aug slots -> random clean once phase 2) ----
    epoch_ref = [0]
    after = args.anneal_after

    class AnnealDS(torch.utils.data.Dataset):
        def __init__(self, base, m_aug):
            self.base, self.m = base, m_aug
            self.n_clean = len(base) - m_aug          # aug are the LAST m rows of tr
            self.rng = np.random.default_rng(42)

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            if epoch_ref[0] >= after and i >= self.n_clean:   # phase 2: aug slot -> clean sample
                i = int(self.rng.integers(0, self.n_clean))
            return self.base[i]

    _orig_build = ft.build_dataset
    _first = [True]

    def _build(*a, **k):
        ds = _orig_build(*a, **k)
        if _first[0]:                                  # the FIRST build_dataset call is train_ds
            _first[0] = False
            logger.info(f"ANNEAL wrap: train len={len(ds)} n_clean={len(ds) - m} m_aug={m}")
            return AnnealDS(ds, m)
        return ds
    ft.build_dataset = _build

    from transformers import Trainer, TrainerCallback

    class AnnealCB(TrainerCallback):
        def __init__(self):
            self.e = -1

        def on_epoch_begin(self, a, s, c, **k):
            self.e += 1
            epoch_ref[0] = self.e
            logger.info(f"ANNEAL epoch {self.e}: {'CLEAN-ONLY' if self.e >= after else 'AUG+CLEAN'}")

    _orig_train = Trainer.train

    def _train(self, *a, **k):
        self.add_callback(AnnealCB())
        return _orig_train(self, *a, **k)
    Trainer.train = _train

    argv = ["src.finetune"] + T070 + [
        "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
        "--max_len", str(args.max_len), "--tag", args.tag,
    ]
    sys.argv = argv
    logger.info(f"E39-anneal tag={args.tag} epochs={args.epochs} anneal_after={after}")
    ft.main()


if __name__ == "__main__":
    main()
