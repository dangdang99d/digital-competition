"""Local before/after evaluation of the test-time-adaptation submission script.

We can't measure TTA's real effect (it depends on train→test shift we can't see),
but we CAN verify locally that the shipped code (a) runs, (b) doesn't COLLAPSE, and
(c) what it costs in wall-time — on the seed-42 clean 25% held-out slice (never seen
by the full_data champion), the same slice run_eval.py --heldout uses.

It imports the EXACT script.py that ships in the zip (via importlib), so the numbers
here are produced by the shipped code, not a reimplementation.

Run (repo root on PYTHONPATH):
  PYTHONPATH=/home/ocean/dacon CUDA_VISIBLE_DEVICES=0 \
    /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/misclf-detection/tta_eval.py \
    --model_dir build_tta/qwen3_tta/model/qwen3-0.6b --variant richargs
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, macro_f1, split_indices


def import_script(path):
    spec = importlib.util.spec_from_file_location("subm_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pred_hist(preds, k=14):
    return np.bincount(preds, minlength=k)


def eval_slice(S, model, encodings, order, remap, collator, device, y_true):
    """Predict with the (possibly adapted) model and score against y_true."""
    t0 = time.perf_counter()
    preds_sorted = S.predict(model, encodings, order, remap, collator, device)
    dt = time.perf_counter() - t0
    preds = np.empty(len(order), dtype=np.int64)
    for pos, orig in enumerate(order):
        preds[orig] = preds_sorted[pos]
    acc = float((preds == y_true).mean())
    f1 = macro_f1(y_true, preds)
    return preds, acc, f1, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="submission model dir (pruned + remap.npy)")
    ap.add_argument("--script", default="build_tta/qwen3_tta/script.py")
    ap.add_argument("--variant", default="richargs")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--full_val", action="store_true",
                    help="use the whole 14k val (default: the clean 25%% held-out slice)")
    ap.add_argument("--configs", default="baked",
                    help="'baked' = ship defaults; or 'sweep' to try a few (lr,steps)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = import_script(args.script)
    from transformers import DataCollatorWithPadding

    # ---- labelled slice: seed-42 val, then the 25% clean held-out (unless --full_val)
    samples, y = load_samples(args.data_dir)
    yid = np.array([CLASS_TO_ID[a] for a in y])
    _, va = split_indices(y, seed=42)
    if args.full_val:
        idx = va
        logger.info(f"eval slice: full seed-42 val ({len(idx)})")
    else:
        _, idx = train_test_split(va, test_size=0.25, stratify=yid[va], random_state=42)
        logger.info(f"eval slice: clean 25% held-out ({len(idx)}) of {len(va)} val")

    sub_samples = [samples[i] for i in idx]
    y_true = yid[idx]
    texts = build_texts(sub_samples, variant=args.variant)

    def fresh_model():
        model, tok, remap, id2label = S.build_model(args.model_dir, device)
        # sanity: model id order must match the project label set, else F1 is garbage
        for i in range(len(ALL_CLASSES)):
            assert id2label[i] == ALL_CLASSES[i], f"label order mismatch at {i}: {id2label[i]}"
        return model, tok, remap

    model, tok, remap = fresh_model()
    collator = DataCollatorWithPadding(tokenizer=tok)
    encodings = S.encode_texts(texts, tok, S.MAX_LENGTH)
    order = sorted(range(len(encodings)),
                   key=lambda i: len(encodings[i]["input_ids"]), reverse=True)

    # ---- baseline (no TTA) ----
    base_preds, base_acc, base_f1, base_dt = eval_slice(
        S, model, encodings, order, remap, collator, device, y_true)
    logger.info(f"BASELINE (no TTA)  acc={base_acc:.4f}  macroF1={base_f1:.4f}  "
                f"predict={base_dt:.1f}s  n={len(y_true)}")
    logger.info(f"  baseline pred hist: {pred_hist(base_preds).tolist()}")

    # ---- config grid ----
    if args.configs == "sweep":
        # characterise the clean-data downside vs adaptation aggressiveness
        grid = [
            dict(lr=2e-4, steps=1, samples=2048, batch=16, lam=1.0),  # gentle
            dict(lr=5e-4, steps=1, samples=2048, batch=16, lam=1.0),
            dict(lr=1e-3, steps=1, samples=1024, batch=16, lam=1.0),  # fewer rows
            dict(lr=1e-3, steps=1, samples=2048, batch=16, lam=1.0),  # tested default
            dict(lr=3e-3, steps=1, samples=2048, batch=16, lam=1.0),  # hot (collapse probe)
        ]
    else:
        grid = [S.tta_config()]   # the baked ship defaults

    print(f"\n{'config':46s} {'acc':>7} {'ΔF1':>8} {'macroF1':>8} {'adapt_s':>8} "
          f"{'ent':>14} {'#argmax_flips':>13}  collapse?")
    print("-" * 118)
    print(f"{'baseline (no TTA)':46s} {base_acc:7.4f} {'—':>8} {base_f1:8.4f} "
          f"{'—':>8} {'—':>14} {'—':>13}")

    for g in grid:
        cfg = S.tta_config()
        cfg.update(g); cfg["enable"] = True
        model, tok, remap = fresh_model()          # reset weights every config
        t0 = time.perf_counter()
        info = S.tent_adapt(model, encodings, order, remap, collator, device, cfg,
                            log=lambda m: None)
        adapt_dt = time.perf_counter() - t0
        preds, acc, f1, _ = eval_slice(S, model, encodings, order, remap, collator, device, y_true)
        flips = int((preds != base_preds).sum())
        h = pred_hist(preds)
        top_frac = h.max() / h.sum()
        collapse = "⚠ COLLAPSE" if top_frac > 0.60 else ""
        tag = (f"lr={g['lr']:g} st={g['steps']} n={g['samples']} "
               f"lam={g['lam']:g} fp32={int(g.get('fp32', False))}")
        ent = f"{info.get('ent_start')}→{info.get('ent_end')}"
        ent = (f"{info['ent_start']:.3f}→{info['ent_end']:.3f}"
               if info.get('ent_start') is not None else "—")
        print(f"{tag:46s} {acc:7.4f} {acc - base_acc:+8.4f} {f1:8.4f} "
              f"{adapt_dt:8.1f} {ent:>14} {flips:13d}  {collapse}")
        if g is grid[-1]:
            logger.info(f"  last-config pred hist: {h.tolist()} (top class {top_frac:.1%})")

    print("\nNotes: ΔF1 vs no-TTA on a CLEAN (no-shift) held-out slice — near-0 is the")
    print("expected/safe outcome here; TTA can only truly help under test-set shift,")
    print("which is not present locally. A large negative ΔF1 or collapse ⇒ too aggressive.")


if __name__ == "__main__":
    main()
