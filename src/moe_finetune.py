"""E33: train the shared-trunk soft-gated GraniteMoE (src/moe_model.py) with the
champion recipe (richargs + LS eps=0.1 + full_data + bf16), from scratch.

Isolated entry point — src/finetune.py is untouched (shared-tree invariant);
data/split/dataset/metrics plumbing is imported from it so the recipe matches
the champion byte-for-byte where it overlaps.

Usage (stage-1 arm, vast 3090):
  python -m src.moe_finetune --full_data --tag e33_moe_k6
Smoke:
  python -m src.moe_finetune --limit 64 --epochs 1 --batch_size 2 --max_len 128

Eval read-outs printed + cached to <run_dir>/moe_val_parts.npz:
  blend / uniform-mean-of-own-experts (gate bypassed — the scientific null) /
  per-expert solos, overall + first-step slice, gate usage/entropy diagnostics.
"""
import argparse
import csv as _csv
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score

from src.data import (ALL_CLASSES, CLASS_TO_ID, SERIALIZE_VARIANTS, build_texts,
                      load_samples, split_indices)
from src.finetune import build_dataset, make_best_snapshot, make_compute_metrics
from src.moe_model import GraniteMoE


class GateFeatsDataset(torch.utils.data.Dataset):
    """Wrap a build_dataset() Dataset, attaching per-row controller scalars."""

    def __init__(self, base_ds, feats):
        assert len(base_ds) == len(feats)
        self.base, self.feats = base_ds, feats

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["gate_feats"] = self.feats[i]
        return item


class GateFeatsCollator:
    """Pop gate_feats (fixed-size floats tokenizer.pad can't handle), pad the
    rest with the base collator, re-attach as a (B, 2) tensor."""

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        feats = [f.pop("gate_feats", None) for f in features]
        batch = self.base(features)
        if feats[0] is not None:
            batch["gate_feats"] = torch.tensor(np.stack(feats), dtype=torch.float32)
        return batch


def gate_feats_for(samples, idx):
    """Structure scalars from the RAW records: [log1p(#history events),
    first-step flag] (the regime axis E1/first-step analysis says matters)."""
    out = np.zeros((len(idx), 2), dtype=np.float32)
    for j, i in enumerate(idx):
        nh = len(samples[i]["history"])
        out[j] = (np.log1p(nh), 1.0 if nh == 0 else 0.0)
    return out


def slice_f1(y_true, y_pred, mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:      # e.g. no first-step rows in a tiny smoke slice
        return float("nan")
    return f1_score(y_true, y_pred, labels=list(range(len(ALL_CLASSES))),
                    average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model",
                    default="ibm-granite/granite-embedding-311m-multilingual-r2")
    ap.add_argument("--serialize", default="richargs",
                    choices=sorted(SERIALIZE_VARIANTS))
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_seed", type=int, default=-1,
                    help="init/shuffle seed; -1 = --seed. Expert classifier e is "
                         "seeded init_seed*1000+e (symmetry breaking — granite "
                         "has zero dropout; see moe_model docstring)")
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16"])
    ap.add_argument("--full_data", action="store_true",
                    help="champion protocol: fold 75%% of val into training, "
                         "eval on the held-out 25%% (full_data CV slice)")
    ap.add_argument("--limit", type=int, default=0, help="cap sizes for smokes")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--trunk_k", type=int, default=6,
                    help="shared trunk depth (granite layers 0..k-1)")
    ap.add_argument("--n_experts", type=int, default=4)
    ap.add_argument("--gate_hidden", type=int, default=128)
    ap.add_argument("--balance_coef", type=float, default=0.01,
                    help="Switch aux coefficient (paper alpha=0.01)")
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--entropy_tau", type=float, default=0.75,
                    help="usage-entropy floor = tau*ln(n_experts)")
    ap.add_argument("--grad_checkpointing", default="off", choices=["on", "off"],
                    help="MoE fits a 3090 without it (~15GB proj); on = VRAM fallback")
    ap.add_argument("--group_by_length", action="store_true")
    ap.add_argument("--truncation_side", default="right", choices=["right", "left"])
    ap.add_argument("--save_dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--results_name", default="moe_results.csv")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)

    init_seed = args.init_seed if args.init_seed >= 0 else args.seed
    set_seed(init_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  model={args.model}  method=moe "
                f"trunk_k={args.trunk_k} n_experts={args.n_experts}")

    # ---- data (mirrors finetune.py's champion path) ----
    samples, y = load_samples(args.data_dir)
    texts = build_texts(samples, input_mode="context", max_hist=None,
                        variant=args.serialize)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    # split on STRING labels exactly like finetune.py:932 — int labels give
    # different stratified splits (the E30 fold-construction trap)
    tr, va = split_indices(y, seed=args.seed)
    if args.full_data:
        from sklearn.model_selection import train_test_split
        va_train, va_eval = train_test_split(
            va, test_size=0.25, stratify=y_ids[va], random_state=args.seed)
        tr = np.concatenate([tr, va_train])
        va = va_eval
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]
    logger.info(f"samples={len(texts)}  train={len(tr)}  val={len(va)}  "
                f"full_data={args.full_data}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.truncation_side = args.truncation_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = GateFeatsDataset(
        build_dataset(tok, [texts[i] for i in tr], y_ids[tr], args.max_len),
        gate_feats_for(samples, tr))
    val_ds = GateFeatsDataset(
        build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len),
        gate_feats_for(samples, va))
    collator = GateFeatsCollator(DataCollatorWithPadding(tok))

    # ---- model: load dense granite fp32, split into trunk + experts + gate ----
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(ALL_CLASSES), torch_dtype=torch.float32,
        attn_implementation="sdpa", trust_remote_code=True,
        id2label={i: c for i, c in enumerate(ALL_CLASSES)},
        label2id={c: i for i, c in enumerate(ALL_CLASSES)},
        ignore_mismatched_sizes=True)
    base.config.pad_token_id = tok.pad_token_id
    model = GraniteMoE(base, trunk_k=args.trunk_k, n_experts=args.n_experts,
                       gate_hidden=args.gate_hidden, init_seed=init_seed,
                       label_smoothing=args.label_smoothing,
                       balance_coef=args.balance_coef,
                       entropy_coef=args.entropy_coef,
                       entropy_tau=args.entropy_tau)
    n_par = sum(p.numel() for p in model.parameters())
    logger.info(f"MoE params: {n_par / 1e6:.0f}M (trunk {args.trunk_k}L + "
                f"{args.n_experts} experts x {model.config.num_hidden_layers - args.trunk_k}L"
                f" + gate)")

    safe = args.model.replace("/", "__") + (f"_{args.tag}" if args.tag else "")
    run_dir = os.path.join(args.out_dir, f"moe_{safe}")
    tok.save_pretrained(run_dir)
    base.config.save_pretrained(run_dir)   # offline skeleton for load_moe()

    if args.precision == "auto":
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    else:
        use_bf16 = device == "cuda" and args.precision == "bf16"
    logger.info(f"precision: {'bf16' if use_bf16 else 'fp16' if device == 'cuda' else 'fp32'}")

    targs = TrainingArguments(
        output_dir=run_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        logging_strategy="steps", logging_steps=50, disable_tqdm=False,
        gradient_checkpointing=(args.grad_checkpointing == "on"),
        group_by_length=args.group_by_length,
        eval_strategy="epoch", save_strategy="no",
        bf16=use_bf16, fp16=(device == "cuda" and not use_bf16),
        report_to="none", seed=init_seed,
        label_names=["labels"], remove_unused_columns=False,
    )
    snap = make_best_snapshot()

    from transformers import TrainerCallback

    class GateStats(TrainerCallback):
        """Log the running gate diagnostics each eval (collapse watch)."""

        def on_evaluate(self, targs_, state, control, model=None, **kw):
            aux = getattr(model, "_last_aux", None)
            if aux:
                logger.info(f"gate: usage={aux['usage']} H={aux['usage_H']:.3f} "
                            f"balance={aux['balance']:.3f} (last train batch)")

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator,
                      compute_metrics=make_compute_metrics(len(ALL_CLASSES)),
                      callbacks=[snap, GateStats()])
    trainer.train()
    if snap.best_state is not None:
        model.load_state_dict(snap.best_state)
        val_f1 = snap.best_f1
        logger.info(f"best-epoch weights (epoch {snap.best_epoch:.1f}) loaded from RAM")
    else:
        val_f1 = trainer.evaluate()["eval_macro_f1"]
    logger.success(f"MoE blend: best val Macro-F1 = {val_f1:.4f}")

    # ---- read-outs: blend vs uniform-null vs solos + gate diagnostics ----
    from torch.utils.data import DataLoader
    model.eval().to(device)
    W, LP = [], []
    amp = torch.autocast(device, dtype=torch.bfloat16,
                         enabled=(device == "cuda" and use_bf16))
    with torch.no_grad(), amp:
        for batch in DataLoader(val_ds, batch_size=args.batch_size * 2,
                                shuffle=False, collate_fn=collator):
            w, lp = model.predict_parts(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["gate_feats"].to(device))
            W.append(w.float().cpu().numpy())
            LP.append(lp.float().cpu().numpy())
    W = np.concatenate(W)                       # (N, E) gate weights
    LP = np.concatenate(LP)                     # (N, E, C) expert log-probs
    P = np.exp(LP)
    yv = y_ids[va]
    zh = np.array([len(samples[i]["history"]) == 0 for i in va])

    blend_pred = (W[:, :, None] * P).sum(1).argmax(-1)
    unif_pred = P.mean(1).argmax(-1)
    res = {
        "blend_f1": slice_f1(yv, blend_pred),
        "uniform_f1": slice_f1(yv, unif_pred),
        "expert_f1": [slice_f1(yv, P[:, e].argmax(-1))
                      for e in range(args.n_experts)],
        "blend_f1_first_step": slice_f1(yv, blend_pred, zh),
        "uniform_f1_first_step": slice_f1(yv, unif_pred, zh),
        "gate_mean_w": W.mean(0).round(4).tolist(),
        "gate_argmax_share": (np.bincount(W.argmax(1), minlength=args.n_experts)
                              / len(W)).round(4).tolist(),
        "gate_row_entropy_mean": float(
            -(W.clip(1e-9) * np.log(W.clip(1e-9))).sum(1).mean()),
        "n_val": int(len(va)), "n_first_step": int(zh.sum()),
    }
    for k, v in res.items():
        logger.info(f"{k}: {v}")
    np.savez(os.path.join(run_dir, "moe_val_parts.npz"),
             gate_w=W, expert_logprobs=LP, labels=yv, first_step=zh,
             va_idx=np.asarray(va))

    # ---- persist + record ----
    save_dt = {"fp16": torch.float16, "bf16": torch.bfloat16,
               "fp32": torch.float32}[args.save_dtype]
    model.save_moe(run_dir, args.model, save_dtype=save_dt,
                   extra={"tag": args.tag, "serialize": args.serialize,
                          "val_macro_f1": round(float(val_f1), 4)})
    logger.info(f"final MoE ({args.save_dtype}) -> {run_dir}")

    row = {"model": args.model, "method": "moe",
           "trunk_k": args.trunk_k, "n_experts": args.n_experts,
           "epochs": args.epochs, "lr": args.lr,
           "val_macro_f1": round(float(val_f1), 4),
           "uniform_f1": round(res["uniform_f1"], 4),
           "expert_f1_best": round(max(res["expert_f1"]), 4),
           "gate_mean_w": json.dumps(res["gate_mean_w"]),
           "tag": args.tag, "max_len": args.max_len,
           "serialize": args.serialize, "full_data": args.full_data,
           "balance_coef": args.balance_coef, "entropy_coef": args.entropy_coef}
    out_csv = os.path.join(args.out_dir, args.results_name)
    os.makedirs(args.out_dir, exist_ok=True)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    logger.info(f"appended -> {out_csv}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
