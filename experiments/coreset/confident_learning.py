"""C1 — Confident Learning (cleanlab) drop-noisy scorer.

Estimates per-train-sample label errors the principled way: k-fold OUT-OF-SAMPLE
predicted probs over the 56k train (train on k-1 folds, predict the held-out fold),
then cleanlab.filter.find_label_issues on (labels, oof_probs). Emits a keep-set
(train minus flagged label errors) + reports the estimated noise rate.

  python experiments/coreset/confident_learning.py \
    --model ibm-granite/granite-embedding-311m-multilingual-r2 --k 4 \
    --out experiments/coreset/keepsets/cleanlab_keep.npy

Long-running (k granite trainings) — launch detached on a free GPU.
"""
import argparse
import os
import tempfile

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.finetune import build_dataset, make_compute_metrics

KEEPDIR = "experiments/coreset/keepsets"


def train_predict(model_name, tok, tr_texts, tr_lab, ev_texts, max_len, epochs, lr, bs, ga):
    import torch
    from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding,
                              Trainer, TrainingArguments)
    id2label = {i: c for i, c in enumerate(ALL_CLASSES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(ALL_CLASSES), id2label=id2label,
        label2id={c: i for i, c in id2label.items()}, trust_remote_code=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    tr_ds = build_dataset(tok, tr_texts, tr_lab, max_len, desc="fold-train")
    ev_ds = build_dataset(tok, ev_texts, np.zeros(len(ev_texts), int), max_len, desc="fold-pred")
    with tempfile.TemporaryDirectory() as td:
        targs = TrainingArguments(
            output_dir=td, num_train_epochs=epochs, learning_rate=lr,
            per_device_train_batch_size=bs, gradient_accumulation_steps=ga,
            bf16=torch.cuda.is_available(), logging_steps=200, report_to=[],
            save_strategy="no", eval_strategy="no")
        tr_trainer = Trainer(model=model, args=targs, train_dataset=tr_ds,
                             data_collator=DataCollatorWithPadding(tok),
                             compute_metrics=make_compute_metrics(len(ALL_CLASSES)))
        tr_trainer.train()
        logits = tr_trainer.predict(ev_ds).predictions
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-embedding-311m-multilingual-r2")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--out", default=f"{KEEPDIR}/cleanlab_keep.npy")
    ap.add_argument("--oof_out", default="analysis/cache/coreset_oof_granite.npz")
    ap.add_argument("--folds", default="", help="comma fold idxs to compute (default all) — "
                    "split across GPUs; pair with --part_out")
    ap.add_argument("--part_out", default="", help="save partial OOF (this GPU's folds) & exit")
    ap.add_argument("--merge", default="", help="comma part npzs -> assemble OOF + cleanlab + keep")
    args = ap.parse_args()
    os.makedirs(KEEPDIR, exist_ok=True)

    from sklearn.model_selection import StratifiedKFold
    from cleanlab.filter import find_label_issues

    samples, y = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, _ = split_indices(y, seed=42)                   # the 56k standard-split train

    if args.merge:                                       # combine partial OOFs -> cleanlab
        oof = np.zeros((len(tr), len(ALL_CLASSES)), dtype=np.float32)
        for pth in args.merge.split(","):
            d = np.load(pth); oof[d["rows"]] = d["probs"]
        assert (oof.sum(1) > 0).all(), "some train rows missing from parts"
        np.savez(args.oof_out, oof=oof, tr=tr, labels=y_ids[tr])
        issues = find_label_issues(labels=y_ids[tr], pred_probs=oof,
                                   return_indices_ranked_by="self_confidence")
        keep = tr[~np.isin(np.arange(len(tr)), issues)]
        np.save(args.out, keep)
        logger.success(f"cleanlab: {len(issues)} issues ({len(issues)/len(tr):.2%}) -> "
                       f"keep {len(keep)}/{len(tr)} -> {args.out}  (OOF acc "
                       f"{np.mean(oof.argmax(1)==y_ids[tr]):.4f})")
        return

    from transformers import AutoTokenizer
    texts = build_texts(samples)                        # v1 (merge doesn't need it)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(tr, y_ids[tr]))
    want = set(int(x) for x in args.folds.split(",")) if args.folds else set(range(args.k))
    oof = np.zeros((len(tr), len(ALL_CLASSES)), dtype=np.float32)
    rows = []
    for fold, (a, b) in enumerate(splits):
        if fold not in want:
            continue
        logger.info(f"=== fold {fold+1}/{args.k}: train {len(a)} / predict {len(b)} ===")
        oof[b] = train_predict(
            args.model, tok, [texts[tr[i]] for i in a], y_ids[tr[a]],
            [texts[tr[i]] for i in b], args.max_len, args.epochs, args.lr,
            args.batch_size, args.grad_accum)
        rows.extend(b.tolist())
    if args.part_out:                                    # this GPU's folds only -> merge later
        rows = np.array(sorted(rows))
        np.savez(args.part_out, rows=rows, probs=oof[rows])
        logger.success(f"partial OOF folds {sorted(want)} -> {args.part_out} ({len(rows)} rows)")
        return
    np.savez(args.oof_out, oof=oof, tr=tr, labels=y_ids[tr])
    issues = find_label_issues(labels=y_ids[tr], pred_probs=oof,
                               return_indices_ranked_by="self_confidence")
    issue_mask = np.zeros(len(tr), bool)
    issue_mask[issues] = True
    keep = tr[~issue_mask]
    np.save(args.out, keep)
    logger.success(f"cleanlab: {issue_mask.sum()} label issues ({issue_mask.mean():.2%}) "
                   f"-> keep {len(keep)}/{len(tr)} -> {args.out}")
    # oof accuracy (sanity: OOF should be well below in-sample)
    logger.info(f"OOF acc {np.mean(oof.argmax(1) == y_ids[tr]):.4f}")


if __name__ == "__main__":
    main()
