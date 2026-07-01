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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="context", choices=["context", "prompt"])
    ap.add_argument("--max_hist", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)          # full-FT needs a small LR
    ap.add_argument("--batch_size", type=int, default=4)       # full-FT is VRAM-heavy (~11GB GPU)
    ap.add_argument("--grad_accum", type=int, default=4)       # effective batch 16
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
    texts = build_texts(samples, input_mode=args.input, max_hist=args.max_hist)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=args.seed)
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]
    logger.info(f"samples={len(texts)}  train={len(tr)}  val={len(va)}")

    # class weights for imbalance (Macro-F1 weights rare classes equally)
    counts = np.bincount(y_ids[tr], minlength=len(ALL_CLASSES)).astype(float)
    class_weights = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))),
                                 dtype=torch.float32, device=device)

    # ---- tokenizer + model + single linear head ----
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(ALL_CLASSES),
        torch_dtype=torch.float32,   # fp32 for stable classifier training
    )
    model.config.pad_token_id = tok.pad_token_id

    # ---- full fine-tuning: ALL backbone weights + head are trainable ----
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable params: {n_trainable:,} / {n_total:,} (100% — full fine-tune)")

    train_ds = build_dataset(tok, [texts[i] for i in tr], y_ids[tr], args.max_len)
    val_ds = build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len)
    collator = DataCollatorWithPadding(tok)

    # class-weighted cross-entropy
    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=class_weights)
            return (loss, out) if return_outputs else loss

    safe = args.model.replace("/", "__")
    run_dir = os.path.join(args.out_dir, f"ft_{safe}")
    targs = TrainingArguments(
        output_dir=run_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_ratio=0.05, weight_decay=0.01,
        logging_strategy="steps", logging_steps=50,   # periodic {loss,epoch} log lines
        disable_tqdm=False,                            # keep the bar; tqdm.auto is log-safe
        gradient_checkpointing=True,                   # trade compute for VRAM (full-FT is heavy)
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, fp16=(device == "cuda"), report_to="none", seed=args.seed,
    )
    trainer = WTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    val_f1 = metrics["eval_macro_f1"]
    logger.success(f"{args.model}: best val Macro-F1 = {val_f1:.4f}")

    # ---- record ----
    os.makedirs(args.out_dir, exist_ok=True)
    row = {"model": args.model, "method": "full_ft", "head": "linear",
           "epochs": args.epochs, "lr": args.lr, "val_macro_f1": round(float(val_f1), 4)}
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
