"""E49: MC-DropHead — turn DropHead ON at inference for K stochastic passes and average
the softmax = a free single-model ensemble (like MC-dropout, but structured/head-level and
now ON-LABEL since the E47 winner trained WITH head-dropout). Scores on the 14k val (the
honest E47 protocol). Compares to the clean single deterministic pass.

Usage (on a DropHead-trained checkpoint):
  python -m experiments.performance-boost.mc_drophead_eval --ckpt <dir> --k 16 --p 0.1
"""
import argparse
import glob
import os

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, macro_f1, split_indices
from src.finetune import install_drophead
from src.runlog import log_cmd


def resolve(d):
    if os.path.exists(os.path.join(d, "model.safetensors")):
        return d
    cks = sorted(glob.glob(os.path.join(d, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    return d


@torch.no_grad()
def forward_probs(model, tok, texts, device, bs=32):
    out = []
    for b in range(0, len(texts), bs):
        enc = tok(texts[b:b + bs], truncation=True, max_length=512,
                  padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits.float(), -1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=16, help="MC passes")
    ap.add_argument("--p", type=float, default=0.1, help="DropHead p at inference")
    ap.add_argument("--variant", default="richargs")
    ap.add_argument("--full_val", action="store_true", help="full 14k val (default: same)")
    ap.add_argument("--out", default="experiments/performance-boost/e48_mc_drophead.csv")
    args = ap.parse_args()
    log_cmd()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples, labels = load_samples("./data")
    yid = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    y = yid[va]
    texts = build_texts([samples[i] for i in va], variant=args.variant)
    logger.info(f"14k val: {len(va)} rows")

    ck = resolve(args.ckpt)
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ck, torch_dtype=torch.float32, trust_remote_code=True).to(device).eval()

    # clean deterministic baseline (no DropHead)
    base = forward_probs(model, tok, texts, device)
    base_f1 = macro_f1(y, base.argmax(1))
    logger.info(f"clean (no MC): macroF1={base_f1:.4f}")

    # install DropHead with inference forced ON, K stochastic passes, average softmax
    state = install_drophead(model, args.p, state={"p": args.p, "infer": True})
    acc = np.zeros_like(base)
    rows = []
    for kk in range(1, args.k + 1):
        torch.manual_seed(1234 + kk)
        acc += forward_probs(model, tok, texts, device)
        if kk in (1, 2, 4, 8, 16, args.k):
            f1 = macro_f1(y, (acc / kk).argmax(1))
            rows.append((kk, f1))
            logger.info(f"  MC K={kk:2d}: macroF1={f1:.4f}  (Δ vs clean {f1 - base_f1:+.4f})")
    best_k, best_f1 = max(rows, key=lambda r: r[1])
    logger.success(f"MC-DropHead p={args.p}: clean {base_f1:.4f} → best {best_f1:.4f} "
                   f"@K={best_k} (Δ {best_f1 - base_f1:+.4f})")
    with open(args.out, "a") as f:
        f.write(f"{ck},p={args.p},clean={base_f1:.4f},bestK={best_k},bestF1={best_f1:.4f},"
                f"delta={best_f1 - base_f1:+.4f}\n")


if __name__ == "__main__":
    main()
