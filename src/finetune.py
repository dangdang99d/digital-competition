"""Fine-tune a backbone with a single linear classification head for the 14-class
next-action task. Metric: Macro-F1 (competition metric).

This run: FULL fine-tuning — all backbone weights + the standard single linear head
are trained (AutoModelForSequenceClassification's built-in head is one linear layer).
Higher ceiling than LoRA but more VRAM/time; uses a small LR (2e-5) and gradient
checkpointing. (The LoRA variant lives on the probe-improvements branch.)

Usage:
  python -m src.finetune --model Qwen/Qwen3-Embedding-0.6B
  python -m src.finetune --model ibm-granite/granite-embedding-311m-multilingual-r2 --epochs 3
Outputs: output/ft_<model>/ (best checkpoint + metrics), appended to output/ft_results.csv.
"""
import argparse
import csv as _csv
import json
import os

# reduce CUDA fragmentation OOMs (must be set before torch initializes CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices


def build_dataset(tok, texts, labels, max_len, desc="tokenizing"):
    """Tokenize once; return a torch Dataset yielding input_ids/attention_mask/labels.

    Tokenizes in chunks with a tqdm bar (works in a terminal and prints periodic
    updates to an sbatch log file).
    """
    from tqdm.auto import tqdm

    enc = {"input_ids": [], "attention_mask": []}
    chunk = 1000
    for i in tqdm(range(0, len(texts), chunk), desc=desc, unit="k-rows",
                  mininterval=5.0):  # mininterval keeps sbatch logs sparse
        e = tok(texts[i:i + chunk], truncation=True, max_length=max_len, padding=False)
        enc["input_ids"].extend(e["input_ids"])
        enc["attention_mask"].extend(e["attention_mask"])

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            return {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": int(labels[i]),
            }

    return DS()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mf1 = f1_score(labels, preds, labels=list(range(len(ALL_CLASSES))),
                   average="macro", zero_division=0)
    return {"macro_f1": mf1}


def _macro_f1(logits, labels, bias):
    preds = np.argmax(logits + bias, axis=1)
    return f1_score(labels, preds, labels=list(range(len(ALL_CLASSES))),
                    average="macro", zero_division=0)


def calibrate_logit_bias(logits, labels, rounds=3, grid=None):
    """Post-hoc per-class logit bias tuned on val to maximize Macro-F1 (73.07 trick).

    Coordinate ascent: for each class, try a grid of additive biases and keep the
    value that improves val Macro-F1. Returns (bias_vector, base_f1, tuned_f1).
    """
    if grid is None:
        grid = np.round(np.arange(-2.0, 2.01, 0.05), 2)
    n = len(ALL_CLASSES)
    bias = np.zeros(n, dtype=np.float32)
    base = _macro_f1(logits, labels, bias)
    best = base
    for _ in range(rounds):
        improved = False
        for c in range(n):
            cur = bias[c]
            best_b, best_f1 = cur, best
            for b in grid:
                bias[c] = b
                f1 = _macro_f1(logits, labels, bias)
                if f1 > best_f1:
                    best_f1, best_b = f1, b
            bias[c] = best_b
            if best_f1 > best:
                best, improved = best_f1, True
        if not improved:
            break
    return bias, base, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="context", choices=["context", "prompt"])
    ap.add_argument("--max_hist", type=int, default=0,
                    help="cap history to last N events; 0 = full history (no cap)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)          # full-FT needs a small LR
    ap.add_argument("--batch_size", type=int, default=4)       # full-FT is VRAM-heavy (~11GB GPU)
    ap.add_argument("--grad_accum", type=int, default=4)       # effective batch 16
    ap.add_argument("--optim", default="adamw_torch",
                    help="optimizer. adamw_torch (default, best quality) or sgd "
                         "(zero optimizer state -> fits bigger models like Qwen3 full-FT). "
                         "SGD usually needs a higher --lr.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--results_name", default="ft_results.csv",
                    help="results CSV filename; give each array task a unique one to "
                         "avoid concurrent-append races")
    ap.add_argument("--limit", type=int, default=0, help="cap train+val size (0=all); for quick tests")
    args = ap.parse_args()

    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  model={args.model}  method=full-finetune+linear-head")

    # ---- data ----
    samples, y = load_samples(args.data_dir)
    max_hist = args.max_hist or None            # 0 -> None (full history)
    texts = build_texts(samples, input_mode=args.input, max_hist=max_hist)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=args.seed)
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]
    logger.info(f"samples={len(texts)}  train={len(tr)}  val={len(va)}  max_hist={max_hist}")

    # ---- tokenizer + model + single linear head ----
    # trust_remote_code: some backbones (e.g. gte's model_type "new") ship custom
    # modeling code and won't load without it.
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(ALL_CLASSES),
        torch_dtype=torch.float32,   # fp32 for stable classifier training
        trust_remote_code=True,
        # real action names so config.id2label maps ids -> actions (not LABEL_0...);
        # needed for logit_bias.json keys and for inference to emit action strings.
        id2label={i: c for i, c in enumerate(ALL_CLASSES)},
        label2id={c: i for i, c in enumerate(ALL_CLASSES)},
        # some backbones ship a pretrained head (e.g. gte has a 1-logit head);
        # discard it and init a fresh 14-class head for our task.
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tok.pad_token_id

    # ---- full fine-tuning: ALL backbone weights + head are trainable ----
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable params: {n_trainable:,} / {n_total:,} (100% — full fine-tune)")

    train_ds = build_dataset(tok, [texts[i] for i in tr], y_ids[tr], args.max_len)
    val_ds = build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len)
    collator = DataCollatorWithPadding(tok)

    # plain cross-entropy (Trainer's default). No class weighting — the 73.07 model
    # handles imbalance via post-hoc logit-bias calibration instead.
    safe = args.model.replace("/", "__")
    run_dir = os.path.join(args.out_dir, f"ft_{safe}")
    targs = TrainingArguments(
        output_dir=run_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        optim=args.optim,                              # sgd for big models (zero optimizer state)
        warmup_ratio=0.05, weight_decay=0.01,
        logging_strategy="steps", logging_steps=50,   # periodic {loss,epoch} log lines
        disable_tqdm=False,                            # keep the bar; tqdm.auto is log-safe
        gradient_checkpointing=True,                   # trade compute for VRAM (full-FT is heavy)
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, fp16=(device == "cuda"), report_to="none", seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    val_f1 = metrics["eval_macro_f1"]
    logger.success(f"{args.model}: best val Macro-F1 = {val_f1:.4f}")

    # ---- logit-bias calibration on val (73.07 trick) ----
    pred_out = trainer.predict(val_ds)
    val_logits = pred_out.predictions
    val_labels = pred_out.label_ids
    bias, base_f1, tuned_f1 = calibrate_logit_bias(val_logits, val_labels)
    logger.success(f"  calibrated: {base_f1:.4f} -> {tuned_f1:.4f} (+{tuned_f1-base_f1:.4f})")
    # save the bias next to the model checkpoint for inference
    id2label = model.config.id2label
    bias_map = {id2label[i]: float(bias[i]) for i in range(len(ALL_CLASSES))}
    with open(os.path.join(run_dir, "logit_bias.json"), "w") as f:
        json.dump({"base_macro_f1": float(base_f1), "tuned_macro_f1": float(tuned_f1),
                   "bias": bias_map}, f, indent=2)

    # ---- record ----
    os.makedirs(args.out_dir, exist_ok=True)
    row = {"model": args.model, "method": "full_ft", "head": "linear",
           "epochs": args.epochs, "lr": args.lr, "val_macro_f1": round(float(val_f1), 4),
           "calibrated_macro_f1": round(float(tuned_f1), 4)}
    out_csv = os.path.join(args.out_dir, args.results_name)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    logger.info(f"appended -> {out_csv}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
