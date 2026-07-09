"""SWA / checkpoint averaging WITHIN one training run.

Averages the weights of the last N epoch checkpoints of a single run (train the
run with --keep_checkpoints N so they exist), saves the averaged model to
<run_dir>/swa/, and evaluates BOTH the best single checkpoint and the average
on the proper eval split — including a freshly tuned logit bias for each — so
the comparison is apples-to-apples.

Scope note: within-run temporal averaging only (the greenlit EMA/SWA idea).
Averaging across independent runs (model soup) was ruled out (Jul 3).

Works for any AutoModelForSequenceClassification run: bge-m3 (XLM-R), Qwen3, etc.

Usage:
  python -m analysis.swa_average --run_dir output/pat/ft_Qwen__Qwen3-Embedding-0.6B_v3_1024_full
  # for a --full_data run, add --full_data so eval uses the same untouched 25% holdout
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, serialize, split_indices
from src.finetune import calibrate_logit_bias


def eval_checkpoint(model_dir, texts, y_true, max_len, state_dict=None, batch_size=16):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model = model.half().cuda().eval()          # fp16 eval: verified prediction-identical
    coll = DataCollatorWithPadding(tokenizer=tok)
    encs = [tok(t, truncation=True, max_length=max_len, padding=False) for t in texts]
    order = sorted(range(len(encs)), key=lambda i: len(encs[i]["input_ids"]), reverse=True)
    logits = np.zeros((len(encs), len(ALL_CLASSES)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            b = {k: v.cuda() for k, v in coll([encs[i] for i in idx]).items()}
            out = model(**b).logits.float().cpu().numpy()
            for j, i in enumerate(idx):
                logits[i] = out[j]
    del model
    torch.cuda.empty_cache()
    bias, base_f1, tuned_f1 = calibrate_logit_bias(logits, y_true)
    return logits, bias, base_f1, tuned_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--last", type=int, default=0, help="checkpoints to average (0 = all present)")
    ap.add_argument("--max_len", type=int, default=512, help="MUST match the run's training max_len")
    ap.add_argument("--full_data", action="store_true",
                    help="run was trained with --full_data: eval on the untouched 25%% of val")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    cks = sorted(glob.glob(os.path.join(args.run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    if args.last:
        cks = cks[-args.last:]
    assert len(cks) >= 2, f"need >=2 checkpoints to average, found {len(cks)} in {args.run_dir}"
    logger.info(f"averaging {len(cks)} checkpoints: {[c.rsplit('/',1)[1] for c in cks]}")

    # ---- running mean over state dicts ----
    from safetensors.torch import load_file
    avg = None
    for k, ck in enumerate(cks):
        sd = load_file(os.path.join(ck, "model.safetensors"))
        if avg is None:
            avg = {n: t.float() for n, t in sd.items()}
        else:
            for n in avg:
                avg[n] += (sd[n].float() - avg[n]) / (k + 1)
    del sd

    # ---- eval split (mirrors finetune.py) ----
    samples, y = load_samples(args.data_dir)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    _, va = split_indices(y, seed=args.seed)
    if args.full_data:
        from sklearn.model_selection import train_test_split
        _, va = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=args.seed)
    texts = [serialize(samples[i]) for i in va]
    y_true = y_ids[va]
    logger.info(f"eval on {len(va)} samples ({'full_data 25% holdout' if args.full_data else 'standard val'})")

    # ---- best single checkpoint (last = best under load_best_model_at_end rotation) ----
    _, b_bias, b_base, b_tuned = eval_checkpoint(cks[-1], texts, y_true, args.max_len)
    logger.info(f"single best ckpt: F1={b_base:.4f}  calibrated={b_tuned:.4f}")

    # ---- SWA average ----
    _, s_bias, s_base, s_tuned = eval_checkpoint(cks[-1], texts, y_true, args.max_len, state_dict=avg)
    logger.info(f"SWA average    : F1={s_base:.4f}  calibrated={s_tuned:.4f}  "
                f"(delta {s_tuned - b_tuned:+.4f})")

    # ---- save the averaged model + its bias next to the run ----
    out_dir = os.path.join(args.run_dir, "swa")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(cks[-1], local_files_only=True,
                                                               torch_dtype=torch.float32)
    model.load_state_dict(avg)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(cks[-1], local_files_only=True).save_pretrained(out_dir)
    id2label = model.config.id2label
    with open(os.path.join(out_dir, "logit_bias.json"), "w") as f:
        json.dump({"base_macro_f1": float(s_base), "tuned_macro_f1": float(s_tuned),
                   "bias": {id2label[i]: float(s_bias[i]) for i in range(len(ALL_CLASSES))}}, f, indent=2)
    logger.success(f"SWA model -> {out_dir}  (use it only if the delta above is positive)")


if __name__ == "__main__":
    main()
