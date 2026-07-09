"""TAPT (task-adaptive continued pretraining) of the granite-311m ModernBERT backbone.

Continues masked-LM pretraining of `ibm-granite/granite-embedding-311m-multilingual-r2`
(base = ModernBERT, a masked LM) on OUR serialized training texts (serialize variant v1,
the exact text the classifier sees), 1-2 epochs, standard 15% MLM. Saves the TAPT'd
encoder to <out_dir>/final for a downstream classification fine-tune via
`src.finetune --init_from <out_dir>/final`.

Method: TAPT/DAPT = Gururangan et al. "Don't Stop Pretraining" (ACL 2020).
Reference implementation: HuggingFace
transformers/examples/pytorch/language-modeling/run_mlm.py — same collator
(DataCollatorForLanguageModeling), masking probability (0.15), and Trainer loop.
Adapted to this repo's custom torch Dataset because the `datasets` library is not
installed in this env (finetune.py's build_dataset follows the same pattern).

Only the classifier TRAIN split (split_indices seed=42, the same split E9 used) is fed
to TAPT, so the downstream classification eval on the held-out val split stays
uncontaminated.

Usage:
  python -m src.mlm_tapt --out_dir ./output/pat/granite_tapt_mlm --epochs 2 \
      --batch_size 16 --lr 5e-5 --max_len 512
"""
import argparse

import torch
from loguru import logger

from src.data import build_texts, load_samples, split_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", default="ibm-granite/granite-embedding-311m-multilingual-r2")
    ap.add_argument("--serialize", default="v1")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./output/pat/granite_tapt_mlm")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"TAPT MLM  device={device}  model={args.model}  serialize={args.serialize}  "
        f"max_len={args.max_len}  bs={args.batch_size}x{args.grad_accum}  lr={args.lr}  "
        f"epochs={args.epochs}  mlm_prob={args.mlm_prob}"
    )

    # ---- corpus: serialized train-split texts, no labels ----
    samples, y = load_samples(args.data_dir)
    tr, va = split_indices(y, seed=args.seed)  # same split as E9's classifier train
    texts_all = build_texts(samples, input_mode="context", variant=args.serialize)
    texts = [texts_all[i] for i in tr]
    logger.info(f"corpus: {len(texts)} serialized train texts (val {len(va)} held out)")

    # ---- tokenizer + masked-LM model ----
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    assert tok.mask_token is not None, "tokenizer has no mask token; MLM impossible"
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
    logger.info(
        f"loaded {model.__class__.__name__}  "
        f"params={sum(p.numel() for p in model.parameters()):,}  "
        f"mask_token={tok.mask_token!r} vocab={len(tok)}"
    )

    # ---- tokenize once (chunked, tqdm; mirrors finetune.build_dataset) ----
    from tqdm.auto import tqdm

    enc = {"input_ids": [], "attention_mask": []}
    chunk = 1000
    for i in tqdm(range(0, len(texts), chunk), desc="tokenizing", unit="k-rows",
                  mininterval=5.0):
        e = tok(texts[i:i + chunk], truncation=True, max_length=args.max_len,
                padding=False)
        enc["input_ids"].extend(e["input_ids"])
        enc["attention_mask"].extend(e["attention_mask"])

    class MLMDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(enc["input_ids"])

        def __getitem__(self, i):
            return {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
            }

    ds = MLMDataset()
    # DataCollatorForLanguageModeling pads the batch and generates the 15%-masked
    # inputs + MLM labels on the fly (canonical run_mlm.py collator).
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=args.mlm_prob)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        bf16=True,
        optim=args.optim,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    final_dir = f"{args.out_dir.rstrip('/')}/final"
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    logger.info(f"TAPT done. Saved TAPT'd encoder to {final_dir}")


if __name__ == "__main__":
    main()
