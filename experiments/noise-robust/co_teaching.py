"""E35 Stage C1 — Co-teaching (Han et al., NeurIPS'18) for the 14-class action task.

= the never-run "Group 3" from E22 (coreset). Two networks train together; each mini-batch
each net keeps the (1-R) LOWEST cross-entropy-loss samples and UPDATES ON THE SAMPLES THE
*OTHER* NET SELECTED (the "exchange"). Because the two nets disagree on which examples look
noisy, they don't reinforce each other's memorization. R(T) ramps 0 -> forget_rate over the
first `num_gradual` epochs (paper's rate_schedule), then stays flat.

Faithful to the official `bhanML/Co-teaching` `loss_coteaching` (loss.py):
  * per-sample CE (reduction='none') -> argsort ascending -> keep num_remember lowest
  * exchange: net1's update loss = CE over net2's kept indices, and vice-versa
The official `noise_or_not` / `pure_ratio` bookkeeping is DROPPED — it needs a ground-truth
clean/noise mask, which exists only for synthetic-noise benchmarks. Our noise is real and
unknown per-row, so pure-ratio is uncomputable (and it never entered the gradient anyway).

RECIPE = the E9/E22 granite CE SCREEN (standard 56k/14k split, v1, from HF base). Co-teaching
uses plain CE internally, so this is a CE-recipe method by construction. Anchors: granite CE
0.7458 (does co-teaching beat plain CE?) and LS 0.7565 (does it beat the champion loss?).
An LS-inside variant is a later probe (E35 gate); this first run stays CE-faithful.

CAVEAT (weaker than CIFAR): the original nets are two RANDOMLY-INIT CNNs. Ours share the same
pretrained granite backbone; divergence comes only from head init + dropout + shuffle order
(--init_seed vs --init_seed2). Two nets off one pretrained backbone are far more correlated
than two random CNNs, so the "the two nets disagree on noise" premise is materially weaker here
-- a real risk that co-teaching degenerates toward a single-net small-loss filter. Reported.

Run (real screen, on an assigned box with a >=16GB GPU; NOT this 8GB local machine):
  python -m experiments.noise-robust.co_teaching \
      --model ibm-granite/granite-embedding-311m-multilingual-r2 \
      --forget_rate 0.20 --num_gradual 5 --epochs 5 \
      --batch_size 32 --lr 2e-5 --warmup_ratio 0.1 --serialize v1 --max_len 512 \
      --init_seed 42 --init_seed2 43 --out output/e35/coteach_fr20.csv
Local smoke (this 4060): add --limit 64 --batch_size 4 --epochs 1 --grad_ckpt on
Forget-rate sweep (E22 over-drop warning: aggressive drop HURT): fr in {0.06, 0.10, 0.20}.
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup,
                          set_seed)
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.finetune import build_dataset
from src.runlog import log_cmd


def build_net(model_name, n_classes, tok, attn_impl, device):
    m = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, torch_dtype=torch.float32,
        attn_implementation=attn_impl, trust_remote_code=True,
        id2label={i: c for i, c in enumerate(ALL_CLASSES)},
        label2id={c: i for i, c in enumerate(ALL_CLASSES)},
        ignore_mismatched_sizes=True,
    )
    m.config.pad_token_id = tok.pad_token_id
    return m.to(device)


def loss_coteaching(logits1, logits2, labels, forget_rate):
    """Faithful port of bhanML/Co-teaching loss.py (minus pure_ratio monitoring).
    Returns the two exchanged update losses (scalars)."""
    per1 = F.cross_entropy(logits1, labels, reduction="none")
    per2 = F.cross_entropy(logits2, labels, reduction="none")
    num_remember = max(1, int(round((1.0 - forget_rate) * labels.shape[0])))
    idx1 = torch.argsort(per1.detach())[:num_remember]   # net1's low-loss picks
    idx2 = torch.argsort(per2.detach())[:num_remember]   # net2's low-loss picks
    # EXCHANGE: net1 updates on net2's picks, net2 on net1's picks
    loss1 = F.cross_entropy(logits1[idx2], labels[idx2])
    loss2 = F.cross_entropy(logits2[idx1], labels[idx1])
    return loss1, loss2


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    preds, gold = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model(**batch).logits
        preds.append(logits.float().argmax(-1).cpu().numpy())
        gold.append(labels.numpy())
    preds = np.concatenate(preds)
    gold = np.concatenate(gold)
    return f1_score(gold, preds, labels=list(range(n_classes)), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-embedding-311m-multilingual-r2")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--serialize", default="v1")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_hist", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42, help="train/val SPLIT seed (fixed)")
    ap.add_argument("--init_seed", type=int, default=42, help="net1 head-init/shuffle seed")
    ap.add_argument("--init_seed2", type=int, default=43, help="net2 head-init/shuffle seed (diversity)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    # co-teaching schedule
    ap.add_argument("--forget_rate", type=float, default=0.20,
                    help="assumed noise rate tau; R(T) ramps 0->tau over num_gradual epochs")
    ap.add_argument("--num_gradual", type=int, default=5, help="Tk: epochs to ramp R to tau")
    ap.add_argument("--exponent", type=float, default=1.0, help="c in R(T)=tau*(T/Tk)^c")
    ap.add_argument("--grad_ckpt", default="off", choices=["on", "off"])
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: cap train/val rows")
    ap.add_argument("--out", default="output/e35/coteach.csv")
    ap.add_argument("--save_dir", default="", help="if set, save the better net's best-epoch weights")
    args = ap.parse_args()
    log_cmd()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(ALL_CLASSES)

    # ---- data (E9/E22 screen: standard split, v1, from HF base) ----
    samples, y = load_samples(args.data_dir)
    max_hist = args.max_hist or None
    texts = build_texts(samples, input_mode="context", max_hist=max_hist, variant=args.serialize)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=args.seed)
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]
    print(f"[co-teaching] train={len(tr)} val={len(va)} forget_rate={args.forget_rate} "
          f"num_gradual={args.num_gradual} epochs={args.epochs}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tr_texts = [texts[i] for i in tr]
    va_texts = [texts[i] for i in va]
    train_ds = build_dataset(tok, tr_texts, y_ids[tr], args.max_len, desc="tok-train")
    val_ds = build_dataset(tok, va_texts, y_ids[va], args.max_len, desc="tok-val")
    collate = DataCollatorWithPadding(tok)

    # net1's shuffle uses init_seed, net2's uses init_seed2 -> different data ORDER too
    g1 = torch.Generator().manual_seed(args.init_seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g1)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    # ---- two nets, two optimizers ----
    set_seed(args.init_seed)
    net1 = build_net(args.model, n_classes, tok, args.attn_impl, device)
    set_seed(args.init_seed2)
    net2 = build_net(args.model, n_classes, tok, args.attn_impl, device)
    if args.grad_ckpt == "on":
        net1.gradient_checkpointing_enable()
        net2.gradient_checkpointing_enable()

    steps = args.epochs * len(train_loader)
    warmup = int(args.warmup_ratio * steps)
    opt1 = torch.optim.AdamW(net1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt2 = torch.optim.AdamW(net2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch1 = get_linear_schedule_with_warmup(opt1, warmup, steps)
    sch2 = get_linear_schedule_with_warmup(opt2, warmup, steps)

    # R(T) schedule (official): linspace(0, tau^c, num_gradual) then flat tau
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
            l1, l2 = loss_coteaching(logits1, logits2, labels, R)
            opt1.zero_grad(set_to_none=True); l1.backward(); opt1.step(); sch1.step()
            opt2.zero_grad(set_to_none=True); l2.backward(); opt2.step(); sch2.step()
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
    print(f"[co-teaching DONE] best_mF1={best:.4f} (net1 {best1:.4f} / net2 {best2:.4f}) "
          f"@epoch {best_epoch}  vs CE 0.7458 (Δ{best-0.7458:+.4f}) / LS 0.7565 (Δ{best-0.7565:+.4f})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["forget_rate", "num_gradual", "epochs", "batch_size", "lr",
                        "best_mF1", "net1_mF1", "net2_mF1", "best_epoch",
                        "d_vs_ce", "d_vs_ls"])
        w.writerow([args.forget_rate, args.num_gradual, args.epochs, args.batch_size, args.lr,
                    f"{best:.4f}", f"{best1:.4f}", f"{best2:.4f}", best_epoch,
                    f"{best-0.7458:+.4f}", f"{best-0.7565:+.4f}"])
    print(f"[co-teaching] wrote {args.out}")


if __name__ == "__main__":
    main()
