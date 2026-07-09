"""coreset_eval.py — richer eval of the drop-noisy retrains than the orchestrator's log-harvest.

For each granite retrain checkpoint (output/pat/ft_..._coreset_*), runs ONE raw forward pass
over the standard-split 14k val and reports, per §3 of results.md:
  overall macro-F1 · per-class F1 · easy/hard-tier macro-F1 (qwen3 ruler) · clean-val macro-F1
  (val minus qwen3-cleanlab suspects) · that keep-set's train drop count & %.
NO logit calibration — raw argmax only (project invariant).

Val hardness ruler = qwen3 champion cached OOF logits (analysis/cache/qwen3_val_logits.npz); the
baseline held val out so on val they are out-of-sample (§3). Two "hard" notions (§3):
  - suspect-mislabel  (qwen3 confidently disagrees with the label)  -> tier F1 is CONFOUNDED
  - ambiguous         (qwen3 low-conf, disagrees)                   -> genuine difficulty
so the meaningful "did dropping help" number is macro-F1 on the CLEAN/EASY subset, not hard-tier F1.

Run AFTER training finishes (GPUs free):
  PYTHONPATH=/home/ocean/dacon /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/coreset/coreset_eval.py
"""
import argparse
import glob
import os
import tempfile

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.finetune import build_dataset

BASELINE_HIST = 0.7458          # E9 granite CE full-56k raw control (historical reference)
KEEPDIR = "experiments/coreset/keepsets"
# tag suffix (after "coreset_rt_") -> keep-set file  (mirror run_coreset.RETRAINS)
KEEPSET = {
    "cl": "cleanlab_keep.npy",
    "cart06": "cart_drophard_drop06.npy", "cart15": "cart_drophard_drop15.npy",
    "aum06": "aum_drop06.npy", "aum15": "aum_drop15.npy",
    "el2n06": "el2n_drop06.npy", "el2n15": "el2n_drop15.npy",
    "forget": "forget_neverlearned.npy",
    "pvi06": "pvi_drop06.npy", "pvi15": "pvi_drop15.npy",
}
CONFUSABLE = ["list_directory", "read_file", "glob_pattern", "grep_search"]  # §1 error-concentrated


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def mf1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def per_class_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, labels=np.arange(len(ALL_CLASSES)), average=None, zero_division=0)


def resolve_ckpt(run_dir):
    """Model weights may sit in run_dir/ OR run_dir/checkpoint-XXXX/ (HF Trainer save)."""
    if os.path.exists(os.path.join(run_dir, "model.safetensors")):
        return run_dir
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
    for c in reversed(cks):                                  # highest step first
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    return None


def predict_val(run_dir, ds, bs):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    tok = AutoTokenizer.from_pretrained(run_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(resolve_ckpt(run_dir), trust_remote_code=True)
    with tempfile.TemporaryDirectory() as td:
        targs = TrainingArguments(output_dir=td, per_device_eval_batch_size=bs, report_to=[],
                                  bf16=torch.cuda.is_available())
        tr = Trainer(model=model, args=targs, data_collator=DataCollatorWithPadding(tok))
        logits = tr.predict(ds).predictions
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def run_tag(run_dir):
    """base 'ft_..._coreset_base' -> 'base'; retrain 'ft_..._coreset_rt_cart06' -> 'cart06'."""
    b = os.path.basename(run_dir.rstrip("/"))
    if b.endswith("_coreset_base"):
        return "base"
    i = b.rfind("_coreset_rt_")
    return b[i + len("_coreset_rt_"):] if i >= 0 else b


def drop_stats(tag, n_tr):
    kf = KEEPSET.get(tag)
    if not kf:
        return 0, 0.0                                   # base / unknown = full data
    p = os.path.join(KEEPDIR, kf)
    if not os.path.exists(p):
        return None, None
    keep = np.load(p)
    drop = n_tr - len(keep)
    return drop, drop / n_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run dirs; default = glob coreset_*")
    ap.add_argument("--ruler", default="analysis/cache/qwen3_val_logits.npz")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--suspect_msp", type=float, default=0.6, help="qwen3 MSP above which a wrong pred = suspect mislabel")
    ap.add_argument("--out", default="experiments/coreset/eval_results.md")
    args = ap.parse_args()

    samples, y = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=42)
    texts = build_texts(samples)                         # v1
    labels_va = y_ids[va]

    # ---- ruler: qwen3 val OOF (ordering sanity via val accuracy) ----
    q = np.load(args.ruler)["logits"]
    assert q.shape[0] == len(va), f"ruler rows {q.shape[0]} != val {len(va)}"
    qp = softmax(q)
    q_pred, q_msp = qp.argmax(1), qp.max(1)
    q_acc = float((q_pred == labels_va).mean())
    logger.info(f"ruler qwen3 val acc {q_acc:.4f} (expect ~0.76 — ordering sanity)")
    assert 0.70 < q_acc < 0.82, f"ruler/val ordering mismatch — qwen3 val acc {q_acc:.4f} off"
    easy = q_pred == labels_va                           # label-trustworthy & model-easy
    hard = ~easy
    suspect = hard & (q_msp > args.suspect_msp)          # qwen3 confident-wrong -> suspect mislabel
    ambig = hard & (q_msp <= args.suspect_msp)           # genuine ambiguity
    from cleanlab.filter import find_label_issues
    cl_issues = find_label_issues(labels=labels_va, pred_probs=qp,
                                  return_indices_ranked_by="self_confidence")
    clean_mask = np.ones(len(va), bool)
    clean_mask[cl_issues] = False                        # val minus cleanlab-on-qwen3 suspects
    logger.info(f"val tiers: easy {easy.sum()} ({easy.mean():.1%}) · hard {hard.sum()} "
                f"[suspect {suspect.sum()} · ambig {ambig.sum()}] · cleanlab-clean {clean_mask.sum()} "
                f"({clean_mask.mean():.1%})")

    # ---- discover finished run dirs ----
    runs = args.runs or sorted(glob.glob("output/pat/ft_*coreset_*"))
    runs = [r for r in runs if resolve_ckpt(r) is not None]
    if not runs:
        logger.warning("no finished coreset run dirs (with model.safetensors) found")
        return
    logger.info(f"evaluating {len(runs)} run(s): {[run_tag(r) for r in runs]}")

    val_ds = None
    rows, pcf = [], {}
    for rd in runs:
        tag = run_tag(rd)
        try:
            if val_ds is None:                           # build once (tokenizer identical across granite runs)
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(rd, trust_remote_code=True)
                val_ds = build_dataset(tok, [texts[i] for i in va], labels_va, args.max_len, desc="val")
            logits = predict_val(rd, val_ds, args.batch_size)
            pred = logits.argmax(1)                       # RAW argmax — no calibration
            drop_n, drop_p = drop_stats(tag, len(tr))
            row = {
                "tag": tag,
                "drop_n": drop_n, "drop_p": drop_p,
                "overall": mf1(labels_va, pred),
                "easy": mf1(labels_va[easy], pred[easy]),
                "hard": mf1(labels_va[hard], pred[hard]),
                "clean": mf1(labels_va[clean_mask], pred[clean_mask]),
                "acc": float((pred == labels_va).mean()),
            }
            rows.append(row)
            pcf[tag] = per_class_f1(labels_va, pred)
            logger.success(f"{tag}: overall {row['overall']:.4f} · easy {row['easy']:.4f} · "
                           f"hard {row['hard']:.4f} · clean {row['clean']:.4f} · "
                           f"drop {drop_n if drop_n is not None else '?'}")
        except Exception as e:
            logger.error(f"{tag}: eval failed — {e}")

    if not rows:
        logger.warning("no runs evaluated successfully")
        return

    base = next((r for r in rows if r["tag"] == "base"), None)
    base_f1 = base["overall"] if base else BASELINE_HIST
    rows.sort(key=lambda r: (r["tag"] != "base", -r["overall"]))   # base first, then best overall

    # ---- write markdown ----
    L = []
    L.append("# Coreset drop-noisy — retrain eval (raw macro-F1, §3 methodology)\n")
    L.append(f"Val = standard-split 14k held-out. Ruler = qwen3 champion OOF (val acc {q_acc:.4f}). "
             f"In-pipeline baseline `base` overall = **{base_f1:.4f}** (historical E9 control 0.7458). "
             "Δ vs `base`. NO calibration.\n")
    L.append(f"Tiers (fixed by qwen3 ruler): easy {easy.sum()} ({easy.mean():.1%}, qwen3-correct → "
             f"label-trustworthy) · hard {hard.sum()} ({hard.mean():.1%}: suspect-mislabel "
             f"{suspect.sum()}, ambiguous {ambig.sum()}) · clean-val {clean_mask.sum()} "
             f"({clean_mask.mean():.1%}, val minus qwen3-cleanlab suspects).\n")
    L.append("**hard-tier F1 is confounded** where labels are suspect — read easy/clean-val for the "
             "'did dropping help' verdict (§3).\n")
    L.append("| run | drop n (%) | overall F1 | Δ base | easy F1 | clean-val F1 | hard F1 | val acc |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        dn = "—" if r["drop_n"] in (None, 0) else f"{r['drop_n']} ({r['drop_p']:.1%})"
        d = "" if r["tag"] == "base" else f"{r['overall']-base_f1:+.4f}"
        L.append(f"| {r['tag']} | {dn} | {r['overall']:.4f} | {d} | {r['easy']:.4f} | "
                 f"{r['clean']:.4f} | {r['hard']:.4f} | {r['acc']:.4f} |")

    # per-class F1 (all 14; confusable flagged with *)
    L.append("\n## Per-class F1 (raw). `*` = §1 error-concentrated (confusable file-ops)\n")
    hdr = [f"{c}{'*' if c in CONFUSABLE else ''}" for c in ALL_CLASSES]
    L.append("| run | " + " | ".join(hdr) + " |")
    L.append("|---|" + "---|" * len(ALL_CLASSES))
    for r in rows:
        v = pcf[r["tag"]]
        L.append(f"| {r['tag']} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    # per-class Δ vs base on the confusable classes (the §1 macro-F1 risk)
    if base:
        bpc = pcf["base"]
        ci = [ALL_CLASSES.index(c) for c in CONFUSABLE]
        L.append("\n## Δ vs base on confusable classes (rare-class guard, §1)\n")
        L.append("| run | " + " | ".join(CONFUSABLE) + " |")
        L.append("|---|" + "---|" * len(CONFUSABLE))
        for r in rows:
            if r["tag"] == "base":
                continue
            v = pcf[r["tag"]]
            L.append(f"| {r['tag']} | " + " | ".join(f"{v[i]-bpc[i]:+.3f}" for i in ci) + " |")

    md = "\n".join(L) + "\n"
    with open(args.out, "w") as f:
        f.write(md)
    logger.success(f"wrote {args.out} ({len(rows)} runs)")
    print("\n" + md)


if __name__ == "__main__":
    main()
