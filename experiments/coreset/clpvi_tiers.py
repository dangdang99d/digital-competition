"""clpvi_tiers.py — CL+PVI val tiering + granite-LS logits per tier (user ask 2026-07-09).

Replace the MSP-heuristic e/a/s ruler with label-audit scores on the 14k val:
  scores from the qwen3 champion cached val logits (val held out of its training ->
  out-of-sample, same source as coreset_eval's clean-val):
    CL  = cleanlab.filter.find_label_issues(val labels, qwen3 val probs)
    PVI = log2 p_qwen3(gold|x) - log2 prior_train(gold)   [bits]
  tiers:
    s (suspect-mislabel) = CL-flagged
    a (ambiguous)        = not flagged & PVI < 0   (input doesn't beat the class prior)
    e (easy)             = not flagged & PVI >= 0

Then one raw fp32 forward pass of the granite-LS standard-split model (e9_granite_ls —
val is out-of-sample; the e8a_ls champion is full_data-trained so its val logits would be
in-sample) over the 14k val, v1 serialization @512. NO calibration anywhere.

Everything saved to analysis/cache/clpvi_tiers_granite_ls.npz:
  logits (14000x14 granite-LS) · labels · va (absolute row idx) · easy/ambig/suspect masks ·
  pvi · cl_flag · qwen3_msp/qwen3_pred (for crosstabs vs the old ruler)

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/ocean/dacon \
    /home/ocean/miniconda3/envs/dacon/bin/python experiments/coreset/clpvi_tiers.py
"""
import argparse
import glob
import os

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

RULER = "analysis/cache/qwen3_val_logits.npz"
OUT = "analysis/cache/clpvi_tiers_granite_ls.npz"
RUN = "output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e9_granite_ls"


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def resolve_ckpt(run_dir):
    if os.path.exists(os.path.join(run_dir, "model.safetensors")):
        return run_dir
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    raise FileNotFoundError(f"no model.safetensors under {run_dir}")


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    samples, y = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=42)
    yv = y_ids[va]
    texts = build_texts(samples)
    txv = [texts[i] for i in va]

    # ---- CL + PVI scores on val from the qwen3 ruler (out-of-sample) ----
    qz = np.load(RULER)["logits"]
    qp = softmax(qz)
    q_pred, q_msp = qp.argmax(1), qp.max(1)
    q_acc = float((q_pred == yv).mean())
    assert 0.70 < q_acc < 0.82, f"ruler ordering mismatch (acc {q_acc:.4f})"

    from cleanlab.filter import find_label_issues
    cl_idx = find_label_issues(labels=yv, pred_probs=qp,
                               return_indices_ranked_by="self_confidence")
    cl_flag = np.zeros(len(yv), bool)
    cl_flag[cl_idx] = True

    prior = np.bincount(y_ids[tr], minlength=len(ALL_CLASSES)) / len(tr)
    g_x = np.clip(qp[np.arange(len(yv)), yv], 1e-6, 1.0)
    pvi = np.log2(g_x) - np.log2(prior[yv] + 1e-12)

    suspect = cl_flag
    ambig = ~cl_flag & (pvi < 0)
    easy = ~cl_flag & (pvi >= 0)
    tiers = {"e": easy, "a": ambig, "s": suspect}
    logger.info("CL+PVI tiers: " + " · ".join(f"{k} {m.sum()} ({m.mean():.1%})"
                                              for k, m in tiers.items()))

    # crosstab vs the old MSP-heuristic ruler (sanity / continuity)
    old = {"easy": q_pred == yv,
           "suspect": (q_pred != yv) & (q_msp > 0.6),
           "ambig": (q_pred != yv) & (q_msp <= 0.6)}
    print("\n| new \\ old | easy | ambig | suspect |")
    print("|---|---|---|---|")
    for k, m in tiers.items():
        print(f"| {k} | {int((m & old['easy']).sum())} | {int((m & old['ambig']).sum())} | "
              f"{int((m & old['suspect']).sum())} |")

    # ---- granite-LS forward pass (raw fp32) ----
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ck = resolve_ckpt(RUN)
    logger.info(f"granite-LS checkpoint: {ck}")
    tok = AutoTokenizer.from_pretrained(RUN, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ck, torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(txv), args.batch_size):
            enc = tok(txv[b:b + args.batch_size], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to("cuda")
            out.append(model(**enc).logits.float().cpu().numpy())
    logits = np.concatenate(out)

    pred = logits.argmax(1)
    logger.success(f"granite-LS val: acc {float((pred == yv).mean()):.4f} · "
                   f"macro-F1 {mf1(yv, pred):.4f}")
    print("\n| tier | n | granite-LS acc | granite-LS macro-F1 |")
    print("|---|---|---|---|")
    for k, m in tiers.items():
        print(f"| {k} | {int(m.sum())} | {float((pred[m] == yv[m]).mean()):.4f} | "
              f"{mf1(yv[m], pred[m]):.4f} |")

    np.savez(OUT, logits=logits, labels=yv, va=va, easy=easy, ambig=ambig, suspect=suspect,
             pvi=pvi, cl_flag=cl_flag, qwen3_msp=q_msp, qwen3_pred=q_pred)
    logger.success(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
