"""E35 Stage C3 — DivideMix (Li et al., ICLR'20). = the never-run "Group 3" from E22.

Two networks. Each epoch: fit a 2-component GMM to per-sample CE loss -> per-row clean probability;
CO-DIVIDE (net A's GMM splits data for net B) into a LABELED (clean) set and an UNLABELED (noisy,
label discarded) set; then MixMatch semi-supervised training — label co-refinement (labeled),
label co-guessing (unlabeled), temperature sharpening, mixup, and a labeled-CE + unlabeled-MSE loss
with a ramped lambda_u + a uniform-prior penalty.

FAITHFUL ports of `LiJunnan1992/DivideMix` (Train_cifar.py): `eval_train` GMM co-divide, `SemiLoss`
(Lx = -mean sum(target*logsoftmax); Lu = mean (softmax-target)^2), `linear_rampup` lambda_u, the
prior KL penalty, and warmup.

⚠️ ONE DOCUMENTED DEVIATION (image -> text): DivideMix mixes INPUTS (`l*img_a+(1-l)*img_b`), which
has no token-id analogue. We use MANIFOLD MIXUP at the penultimate layer instead: because the final
classifier is LINEAR, mixing the pre-classifier feature is exactly mixing LOGITS
(W(l*h_a+(1-l)*h_b) = l*logits_a+(1-l)*logits_b), so we mix in LOGIT space — exact penultimate
manifold mixup for this architecture, no model surgery. The two augmented views (inputs_x/inputs_x2)
become two DROPOUT views (two train-mode forwards). These are the standard NLP-DivideMix adaptations;
flagged per the "get official code first / note divergences" invariant.

Same recipe/anchors/caveats as co_teaching.py (CE screen, standard split). Heaviest arm: per epoch =
2 GMM eval passes + MixMatch on both nets (2 views + mixup) ~= 4-6x a single run; needs a >=16GB GPU.
The two-net-diversity caveat (shared pretrained backbone) applies here too.

Run:
  python experiments/noise-robust/dividemix.py \
    --model ibm-granite/granite-embedding-311m-multilingual-r2 \
    --p_threshold 0.5 --T 0.5 --alpha 4 --lambda_u 25 --warmup 2 --epochs 6 \
    --batch_size 16 --grad_ckpt on --lr 2e-5 --serialize v1 --max_len 512 \
    --out output/e35/dividemix.csv
Sweep: lambda_u in {1,25} (25 = image default, likely too strong for ~20% low-noise text),
p_threshold in {0.5,0.6}. Local smoke (CPU): --limit 40 --batch_size 4 --warmup 1 --epochs 2.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup, set_seed)

sys.path.insert(0, os.path.dirname(__file__))
from co_teaching import build_net, evaluate            # noqa: E402  (shared helpers)

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.finetune import build_dataset
from src.runlog import log_cmd


def eval_train(net, loader, n_train, device):
    """Per-sample CE over the whole train set -> min-max normalize -> 2-comp GMM ->
    P(clean) = posterior of the lower-mean (small-loss) component. (DivideMix eval_train)"""
    net.eval()
    losses = torch.zeros(n_train)
    with torch.no_grad():
        for batch in loader:
            index = batch.pop("index")
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = net(**batch).logits.float()
            per = F.cross_entropy(logits, labels, reduction="none").cpu()
            losses[index] = per
    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-12)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(losses.reshape(-1, 1).numpy())
    prob = gmm.predict_proba(losses.reshape(-1, 1).numpy())
    return prob[:, gmm.means_.argmin()]          # clean probability per train row


def index_collate(tok):
    base = DataCollatorWithPadding(tok)

    def collate(features):
        idx = torch.tensor([f.pop("index") for f in features], dtype=torch.long)
        batch = base(features)
        batch["index"] = idx
        return batch
    return collate


class IndexDS(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        item["index"] = i
        return item


def linear_rampup(current, warm_up, lambda_u, rampup_len=16):
    return lambda_u * float(np.clip((current - warm_up) / rampup_len, 0.0, 1.0))


def forward_logits(net, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        return net(**batch).logits.float()


def train_divide(epoch, warm_up, net, net2, opt, sch, labeled_loader, unlabeled_loader,
                 clean_prob, n_classes, args, device):
    """Train `net` (net2 fixed, eval) on the CO-DIVIDE of the OTHER net. MixMatch with
    logit-space (penultimate manifold) mixup + dropout views."""
    net.train(); net2.eval()
    u_iter = iter(unlabeled_loader)
    for lb in labeled_loader:
        try:
            ub = next(u_iter)
        except StopIteration:
            u_iter = iter(unlabeled_loader); ub = next(u_iter)
        idx_x = lb.pop("index"); labels_x = lb.pop("labels")
        ub.pop("index"); ub.pop("labels")
        bs = labels_x.shape[0]
        oh_x = F.one_hot(labels_x, n_classes).float().to(device)
        w_x = torch.tensor(clean_prob[idx_x.numpy()], dtype=torch.float32, device=device).view(-1, 1)

        with torch.no_grad():
            # co-guess unlabeled (2 dropout views x 2 nets), sharpen
            lu11 = forward_logits(net, ub, device); lu12 = forward_logits(net, ub, device)
            lu21 = forward_logits(net2, ub, device); lu22 = forward_logits(net2, ub, device)
            pu = (F.softmax(lu11, 1) + F.softmax(lu12, 1) + F.softmax(lu21, 1) + F.softmax(lu22, 1)) / 4
            ptu = pu ** (1 / args.T)
            targets_u = (ptu / ptu.sum(1, keepdim=True)).detach()
            # co-refine labeled (2 dropout views of net), blend with given label by w_x, sharpen
            lx1 = forward_logits(net, lb, device); lx2 = forward_logits(net, lb, device)
            px = (F.softmax(lx1, 1) + F.softmax(lx2, 1)) / 2
            px = w_x * oh_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)
            targets_x = (ptx / ptx.sum(1, keepdim=True)).detach()

        # forward the grad views, then MIX IN LOGIT SPACE (== penultimate manifold mixup)
        gx1 = forward_logits(net, lb, device); gx2 = forward_logits(net, lb, device)
        gu1 = forward_logits(net, ub, device); gu2 = forward_logits(net, ub, device)
        all_logits = torch.cat([gx1, gx2, gu1, gu2], 0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], 0)
        m = min(all_logits.shape[0], all_targets.shape[0])   # views can differ if last batch short
        all_logits, all_targets = all_logits[:m], all_targets[:m]

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        perm = torch.randperm(all_logits.shape[0], device=device)
        mixed_logits = l * all_logits + (1 - l) * all_logits[perm]
        mixed_targets = l * all_targets + (1 - l) * all_targets[perm]

        n_lab = min(2 * bs, mixed_logits.shape[0])
        logits_x, tgt_x = mixed_logits[:n_lab], mixed_targets[:n_lab]
        logits_u, tgt_u = mixed_logits[n_lab:], mixed_targets[n_lab:]
        Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, 1) * tgt_x, 1))
        Lu = torch.mean((F.softmax(logits_u, 1) - tgt_u) ** 2) if logits_u.shape[0] > 0 else logits_x.sum() * 0
        lamb = linear_rampup(epoch, warm_up, args.lambda_u)
        # uniform-prior penalty (anti-collapse)
        prior = torch.full((n_classes,), 1.0 / n_classes, device=device)
        pred_mean = F.softmax(all_logits, 1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / (pred_mean + 1e-8)))
        loss = Lx + lamb * Lu + penalty
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()


def warmup(net, opt, sch, loader, device):
    net.train()
    for batch in loader:
        batch.pop("index")
        labels = batch.pop("labels").to(device)
        logits = forward_logits(net, batch, device)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()


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
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2, help="CE warmup epochs before co-divide")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--p_threshold", type=float, default=0.5, help="clean-prob cutoff -> labeled set")
    ap.add_argument("--T", type=float, default=0.5, help="sharpening temperature")
    ap.add_argument("--alpha", type=float, default=4.0, help="Beta(alpha,alpha) mixup")
    ap.add_argument("--lambda_u", type=float, default=25.0, help="unlabeled MSE weight (ramped)")
    ap.add_argument("--grad_ckpt", default="off", choices=["on", "off"])
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="output/e35/dividemix.csv")
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
    n_train = len(tr)
    print(f"[dividemix] train={n_train} val={len(va)} warmup={args.warmup} epochs={args.epochs} "
          f"lambda_u={args.lambda_u} p_thr={args.p_threshold}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    labels_tr = y_ids[tr]
    train_ds = IndexDS(build_dataset(tok, [texts[i] for i in tr], labels_tr, args.max_len, desc="tok-train"))
    val_ds = build_dataset(tok, [texts[i] for i in va], y_ids[va], args.max_len, desc="tok-val")
    coll = index_collate(tok)
    eval_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=False, collate_fn=coll)
    warm_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False,
                                             collate_fn=DataCollatorWithPadding(tok))

    set_seed(args.init_seed); net1 = build_net(args.model, n_classes, tok, args.attn_impl, device)
    set_seed(args.init_seed2); net2 = build_net(args.model, n_classes, tok, args.attn_impl, device)
    if args.grad_ckpt == "on":
        net1.gradient_checkpointing_enable(); net2.gradient_checkpointing_enable()

    steps = args.epochs * (n_train // args.batch_size + 1)
    opt1 = torch.optim.AdamW(net1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt2 = torch.optim.AdamW(net2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch1 = get_linear_schedule_with_warmup(opt1, int(args.warmup_ratio * steps), steps)
    sch2 = get_linear_schedule_with_warmup(opt2, int(args.warmup_ratio * steps), steps)

    def make_split_loaders(clean_prob):
        clean = np.where(clean_prob > args.p_threshold)[0]
        noisy = np.where(clean_prob <= args.p_threshold)[0]
        if len(clean) == 0:
            clean = np.array([int(clean_prob.argmax())])
        if len(noisy) == 0:
            noisy = np.array([int(clean_prob.argmin())])
        lab = torch.utils.data.DataLoader(torch.utils.data.Subset(train_ds, clean.tolist()),
                                          batch_size=args.batch_size, shuffle=True, collate_fn=coll)
        unl = torch.utils.data.DataLoader(torch.utils.data.Subset(train_ds, noisy.tolist()),
                                          batch_size=args.batch_size, shuffle=True, collate_fn=coll)
        return lab, unl, len(clean), len(noisy)

    best1 = best2 = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        if epoch < args.warmup:
            warmup(net1, opt1, sch1, warm_loader, device)
            warmup(net2, opt2, sch2, warm_loader, device)
            phase = "warmup"
        else:
            prob1 = eval_train(net1, eval_loader, n_train, device)
            prob2 = eval_train(net2, eval_loader, n_train, device)
            # co-divide: net1 trains on net2's split, net2 on net1's split
            lab2, unl2, nc2, _ = make_split_loaders(prob2)
            train_divide(epoch, args.warmup, net1, net2, opt1, sch1, lab2, unl2, prob2, n_classes, args, device)
            lab1, unl1, nc1, _ = make_split_loaders(prob1)
            train_divide(epoch, args.warmup, net2, net1, opt2, sch2, lab1, unl1, prob1, n_classes, args, device)
            phase = f"divide(clean n1={nc1} n2={nc2})"
        f1_1 = evaluate(net1, val_loader, device, n_classes)
        f1_2 = evaluate(net2, val_loader, device, n_classes)
        print(f"[epoch {epoch+1}/{args.epochs}] {phase}  net1_mF1={f1_1:.4f}  net2_mF1={f1_2:.4f}")
        if max(f1_1, f1_2) > max(best1, best2):
            best_epoch = epoch + 1
            if args.save_dir:
                (net1 if f1_1 >= f1_2 else net2).save_pretrained(args.save_dir); tok.save_pretrained(args.save_dir)
        best1, best2 = max(best1, f1_1), max(best2, f1_2)

    best = max(best1, best2)
    print(f"[dividemix DONE] best_mF1={best:.4f} (net1 {best1:.4f} / net2 {best2:.4f}) @epoch {best_epoch}  "
          f"vs CE 0.7458 (Δ{best-0.7458:+.4f}) / LS 0.7565 (Δ{best-0.7565:+.4f})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["lambda_u", "p_threshold", "T", "alpha", "warmup", "epochs", "batch_size",
                        "best_mF1", "net1_mF1", "net2_mF1", "best_epoch", "d_vs_ce", "d_vs_ls"])
        w.writerow([args.lambda_u, args.p_threshold, args.T, args.alpha, args.warmup, args.epochs,
                    args.batch_size, f"{best:.4f}", f"{best1:.4f}", f"{best2:.4f}", best_epoch,
                    f"{best-0.7458:+.4f}", f"{best-0.7565:+.4f}"])
    print(f"[dividemix] wrote {args.out}")


if __name__ == "__main__":
    main()
