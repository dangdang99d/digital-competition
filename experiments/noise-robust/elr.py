"""E35 Stage B — ELR (Early-Learning Regularization, Liu et al., NeurIPS'20).

Single network + a per-sample EMA of its own softmax ("target"). The regularizer steers the
current prediction toward that early-formed target and cancels the gradient on samples the net
already disagrees with -- exploiting that the net fits the CLEAN labels early, before memorizing
noise. Loss = CE(or LS-CE) + lambda * mean( log(1 - <target[i], softmax_i>) ), with
target[i] <- beta*target[i] + (1-beta)*normalize(softmax_i.detach()).

Faithful to `shengliu66/ELR` (ELR/model/loss.py `elr_loss.forward`): softmax clamp [1e-4, 1-1e-4],
row-normalized EMA target, reg = ((1-(target*y_pred).sum(1)).log()).mean(). The only change is a
`.clamp_min(1e-7)` inside the log for fp-safety (mathematically inert; the argument is already >0).

WHY ELR IS THE "+LS" EXCEPTION (E35 §"LS or not?"): the Stage-A robust losses REPLACE the CE/target
term (so does LS -> they're rivals -> trained standalone). ELR ADDS an orthogonal regularizer, so it
genuinely stacks with LS. The deployment-natural form is LS-CE + lambda*ELR-reg -> `--ce_mode ls`
(default here). `--ce_mode ce` (vs CE 0.7458) confirms the reg does anything at all.

RECIPE = the E9/E22 granite screen (standard 56k/14k split, v1, from HF base). Anchors: CE 0.7458
(--ce_mode ce) / LS 0.7565 (--ce_mode ls, the primary). Single model -> fits an 8GB card; this file
CAN smoke-test on the local 4060 (unlike the two-model Stage C).

Run (real screen):
  python experiments/noise-robust/elr.py \
    --model ibm-granite/granite-embedding-311m-multilingual-r2 \
    --ce_mode ls --elr_lambda 3 --elr_beta 0.7 --epochs 5 \
    --batch_size 32 --lr 2e-5 --warmup_ratio 0.1 --serialize v1 --max_len 512 \
    --out output/e35/elr_ls_l3_b07.csv
Sweep: elr_lambda in {1,3,7}, elr_beta in {0.7,0.9}; ce_mode in {ls (primary), ce}.
Local smoke: add --limit 64 --batch_size 4 --epochs 1
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup, set_seed)

sys.path.insert(0, os.path.dirname(__file__))          # so `import co_teaching` resolves
from co_teaching import build_net, evaluate            # noqa: E402  (shared helpers)

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.finetune import build_dataset, classification_loss, awp_perturb, awp_restore
from src.runlog import log_cmd


def make_index_collate(tok):
    base = DataCollatorWithPadding(tok)

    def collate(features):
        idx = torch.tensor([f.pop("index") for f in features], dtype=torch.long)
        batch = base(features)
        batch["index"] = idx
        return batch

    return collate


class IndexDS(torch.utils.data.Dataset):
    """Wrap a base dataset so each item carries its train-row position (for the EMA target)."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        item["index"] = i
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-embedding-311m-multilingual-r2")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--serialize", default="v1")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_hist", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    # ELR
    ap.add_argument("--ce_mode", default="ls", choices=["ce", "ls"],
                    help="base term: ls = LS-CE (primary, ELR is additive to LS) / ce")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--elr_lambda", type=float, default=3.0, help="ELR reg strength")
    ap.add_argument("--elr_beta", type=float, default=0.7, help="EMA momentum for the target")
    # AWP (E34-C) — combine the two best training-time levers; reuses src.finetune.awp_perturb
    ap.add_argument("--awp_gamma", type=float, default=0.0,
                    help="AWP weight-perturbation box (E34 winner: 1e-3; 0 = off)")
    ap.add_argument("--awp_lr", type=float, default=1e-4, help="AWP relative step size (E34: 1e-4)")
    ap.add_argument("--awp_start_epoch", type=float, default=1.0,
                    help="enable AWP once epoch index reaches this (E34: 1.0 = from 2nd epoch)")
    ap.add_argument("--grad_ckpt", default="off", choices=["on", "off"])
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full_data", action="store_true",
                    help="champion recipe: grow train with 75%% of val, eval on the 25%% "
                         "held-out (mirrors src/finetune.py --full_data exactly; full_data CV)")
    ap.add_argument("--out", default="output/e35/elr.csv")
    ap.add_argument("--save_dir", default="")
    args = ap.parse_args()
    log_cmd()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(ALL_CLASSES)
    ls_eps = args.label_smoothing if args.ce_mode == "ls" else 0.0

    samples, y = load_samples(args.data_dir)
    max_hist = args.max_hist or None
    texts = build_texts(samples, input_mode="context", max_hist=max_hist, variant=args.serialize)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=args.seed)
    if args.full_data:
        # mirror src/finetune.py --full_data EXACTLY (same random_state) so the eval
        # slice matches the champion E8a+LS run: train += 75% of val, eval on the 25%
        from sklearn.model_selection import train_test_split
        va_train, va_eval = train_test_split(
            va, test_size=0.25, stratify=y_ids[va], random_state=args.seed)
        tr = np.concatenate([tr, va_train])
        va = va_eval
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]
    print(f"[elr] train={len(tr)} val={len(va)} ce_mode={args.ce_mode} "
          f"lambda={args.elr_lambda} beta={args.elr_beta} epochs={args.epochs}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = IndexDS(build_dataset(tok, [texts[i] for i in tr], y_ids[tr], args.max_len, desc="tok-train"))
    val_ds = build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len, desc="tok-val")

    set_seed(args.init_seed)
    g = torch.Generator().manual_seed(args.init_seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_index_collate(tok), generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=DataCollatorWithPadding(tok))

    net = build_net(args.model, n_classes, tok, args.attn_impl, device)
    if args.grad_ckpt == "on":
        net.gradient_checkpointing_enable()
    # per-sample EMA target buffer (shengliu66/ELR): one prob vector per TRAIN row
    target = torch.zeros(len(tr), n_classes, device=device)

    steps = args.epochs * len(train_loader)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * steps), steps)

    best = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        net.train()
        for batch in train_loader:
            index = batch.pop("index").to(device)
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = net(**batch).logits.float()
            ce = classification_loss(logits, labels, args.ce_mode, label_smoothing=ls_eps)
            y_pred = F.softmax(logits, dim=1).clamp(1e-4, 1.0 - 1e-4)
            y_det = y_pred.detach()
            target[index] = args.elr_beta * target[index] + (1 - args.elr_beta) * (y_det / y_det.sum(1, keepdim=True))
            elr_reg = ((1.0 - (target[index] * y_pred).sum(1)).clamp_min(1e-7).log()).mean()
            loss = ce + args.elr_lambda * elr_reg
            opt.zero_grad(set_to_none=True); loss.backward()
            # AWP: perturb weights -> adversarial forward+backward (grads ADD) -> restore.
            # Target EMA already updated from the CLEAN pass; the adv pass only READS it.
            if args.awp_gamma > 0 and epoch >= args.awp_start_epoch:
                backup = awp_perturb(net, args.awp_lr, args.awp_gamma)
                if backup:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                        adv_logits = net(**batch).logits.float()
                    adv_ce = classification_loss(adv_logits, labels, args.ce_mode, label_smoothing=ls_eps)
                    adv_pred = F.softmax(adv_logits, dim=1).clamp(1e-4, 1.0 - 1e-4)
                    adv_reg = ((1.0 - (target[index] * adv_pred).sum(1)).clamp_min(1e-7).log()).mean()
                    (adv_ce + args.elr_lambda * adv_reg).backward()
                    awp_restore(net, backup)
            opt.step(); sch.step()
        f1 = evaluate(net, val_loader, device, n_classes)
        print(f"[epoch {epoch+1}/{args.epochs}] mF1={f1:.4f}")
        best = max(best, f1)          # tracked for reference only; NOT the reported number

    # FIXED to the FINAL epoch (the full_data 3.5k slice is too noisy for reliable best-epoch
    # selection — user 2026-07-13). The last epoch is the reported result; save its weights.
    final_f1 = f1
    if args.save_dir:
        net.save_pretrained(args.save_dir); tok.save_pretrained(args.save_dir)
    print(f"[elr DONE] final_mF1={final_f1:.4f} @epoch {args.epochs} (best-any {best:.4f})  "
          f"vs CE 0.7458 (Δ{final_f1-0.7458:+.4f}) / LS 0.7565 (Δ{final_f1-0.7565:+.4f})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ce_mode", "elr_lambda", "elr_beta", "epochs", "batch_size", "lr",
                        "final_mF1", "best_any_mF1", "d_vs_ce", "d_vs_ls"])
        w.writerow([args.ce_mode, args.elr_lambda, args.elr_beta, args.epochs, args.batch_size,
                    args.lr, f"{final_f1:.4f}", f"{best:.4f}",
                    f"{final_f1-0.7458:+.4f}", f"{final_f1-0.7565:+.4f}"])
    print(f"[elr] wrote {args.out}")


if __name__ == "__main__":
    main()
