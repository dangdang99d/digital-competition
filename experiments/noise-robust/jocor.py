"""E35 Stage C2 — JoCoR (Wei et al., CVPR'20). = the never-run "Group 3" from E22.

Co-teaching's cousin: two nets, but instead of the EXCHANGE, JoCoR trains them JOINTLY with a
co-regularization (agreement) term and selects the small-loss samples by the JOINT loss, so both
nets update on the SAME co-selected clean set and are pulled toward agreement. One optimizer over
both nets' parameters.

Faithful to `hongxin001/JoCoR` (algorithm/loss.py `loss_jocor`):
  loss_pick_i = (1-co_lambda) * CE(y_i, t)                             # per-sample
  loss_pick   = loss_pick_1 + loss_pick_2 + co_lambda*(KL(y1||y2) + KL(y2||y1))
  keep the (1-R) lowest-loss_pick samples;  loss = mean(loss_pick[kept])  -> one backward, both nets
KL(p||q) = sum_k softmax(q)_k-weighted log-ratio via F.kl_div(log_softmax(p), softmax(q)); neither
side detached (the agreement gradient flows to both). R(T) ramps 0->forget_rate over num_gradual
epochs, as in co-teaching. The official `noise_or_not`/`pure_ratio` monitor is dropped (real noise).

Same recipe/anchors/caveats as co_teaching.py (CE screen, standard split; two nets off ONE
pretrained granite backbone are more correlated than CIFAR's random CNNs -> weaker diversity, the
agreement term may just accelerate collapse to a single-net filter). Needs a >=16GB GPU (two nets).

Run:
  python experiments/noise-robust/jocor.py \
    --model ibm-granite/granite-embedding-311m-multilingual-r2 \
    --forget_rate 0.10 --co_lambda 0.1 --num_gradual 5 --epochs 5 \
    --batch_size 24 --grad_ckpt on --lr 2e-5 --serialize v1 --max_len 512 \
    --out output/e35/jocor_fr10.csv
Sweep: forget_rate in {0.06,0.10,0.20}, co_lambda in {0.05,0.1,0.9} (0.9 = agreement-dominant).
Local smoke (CPU): --limit 24 --batch_size 4 --epochs 1 (two nets OOM the 8GB GPU).
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

sys.path.insert(0, os.path.dirname(__file__))
from co_teaching import build_net, evaluate            # noqa: E402  (shared helpers)

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.finetune import build_dataset
from src.runlog import log_cmd


def kl_per_sample(pred, soft_targets):
    """sum_k KL(log_softmax(pred) || softmax(soft_targets)), per sample (hongxin001/JoCoR)."""
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduction="none")
    return kl.sum(dim=1)


def loss_jocor(logits1, logits2, labels, forget_rate, co_lambda):
    ce1 = F.cross_entropy(logits1, labels, reduction="none") * (1.0 - co_lambda)
    ce2 = F.cross_entropy(logits2, labels, reduction="none") * (1.0 - co_lambda)
    loss_pick = ce1 + ce2 + co_lambda * (kl_per_sample(logits1, logits2)
                                         + kl_per_sample(logits2, logits1))
    num_remember = max(1, int(round((1.0 - forget_rate) * labels.shape[0])))
    keep = torch.argsort(loss_pick.detach())[:num_remember]
    return loss_pick[keep].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-embedding-311m-multilingual-r2")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--serialize", default="v1")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_hist", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_seed", type=int, default=42)
    ap.add_argument("--init_seed2", type=int, default=43)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--forget_rate", type=float, default=0.10)
    ap.add_argument("--co_lambda", type=float, default=0.1, help="agreement (co-reg) weight")
    ap.add_argument("--num_gradual", type=int, default=5)
    ap.add_argument("--exponent", type=float, default=1.0)
    ap.add_argument("--grad_ckpt", default="off", choices=["on", "off"])
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="output/e35/jocor.csv")
    ap.add_argument("--save_dir", default="")
    args = ap.parse_args()
    log_cmd()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(ALL_CLASSES)

    samples, y = load_samples(args.data_dir)
    max_hist = args.max_hist or None
    texts = build_texts(samples, input_mode="context", max_hist=max_hist, variant=args.serialize)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=args.seed)
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]
    print(f"[jocor] train={len(tr)} val={len(va)} forget_rate={args.forget_rate} "
          f"co_lambda={args.co_lambda} num_gradual={args.num_gradual} epochs={args.epochs}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = build_dataset(tok, [texts[i] for i in tr], y_ids[tr], args.max_len, desc="tok-train")
    val_ds = build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len, desc="tok-val")
    collate = DataCollatorWithPadding(tok)
    g = torch.Generator().manual_seed(args.init_seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    set_seed(args.init_seed)
    net1 = build_net(args.model, n_classes, tok, args.attn_impl, device)
    set_seed(args.init_seed2)
    net2 = build_net(args.model, n_classes, tok, args.attn_impl, device)
    if args.grad_ckpt == "on":
        net1.gradient_checkpointing_enable(); net2.gradient_checkpointing_enable()

    steps = args.epochs * len(train_loader)
    # ONE optimizer over BOTH nets (JoCoR: joint loss)
    opt = torch.optim.AdamW(list(net1.parameters()) + list(net2.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    sch = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * steps), steps)

    rate = np.ones(args.epochs) * args.forget_rate
    ng = min(args.num_gradual, args.epochs)
    rate[:ng] = np.linspace(0, args.forget_rate ** args.exponent, ng)

    best1 = best2 = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        R = float(rate[epoch])
        net1.train(); net2.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits1 = net1(**batch).logits.float()
                logits2 = net2(**batch).logits.float()
            loss = loss_jocor(logits1, logits2, labels, R, args.co_lambda)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        f1_1 = evaluate(net1, val_loader, device, n_classes)
        f1_2 = evaluate(net2, val_loader, device, n_classes)
        print(f"[epoch {epoch+1}/{args.epochs}] R={R:.3f}  net1_mF1={f1_1:.4f}  net2_mF1={f1_2:.4f}")
        if max(f1_1, f1_2) > max(best1, best2):
            best_epoch = epoch + 1
            if args.save_dir:
                (net1 if f1_1 >= f1_2 else net2).save_pretrained(args.save_dir)
                tok.save_pretrained(args.save_dir)
        best1, best2 = max(best1, f1_1), max(best2, f1_2)

    best = max(best1, best2)
    print(f"[jocor DONE] best_mF1={best:.4f} (net1 {best1:.4f} / net2 {best2:.4f}) @epoch {best_epoch}  "
          f"vs CE 0.7458 (Δ{best-0.7458:+.4f}) / LS 0.7565 (Δ{best-0.7565:+.4f})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["forget_rate", "co_lambda", "num_gradual", "epochs", "batch_size", "lr",
                        "best_mF1", "net1_mF1", "net2_mF1", "best_epoch", "d_vs_ce", "d_vs_ls"])
        w.writerow([args.forget_rate, args.co_lambda, args.num_gradual, args.epochs, args.batch_size,
                    args.lr, f"{best:.4f}", f"{best1:.4f}", f"{best2:.4f}", best_epoch,
                    f"{best-0.7458:+.4f}", f"{best-0.7565:+.4f}"])
    print(f"[jocor] wrote {args.out}")


if __name__ == "__main__":
    main()
