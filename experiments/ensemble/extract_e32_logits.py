"""Extract E32 augmentation models' held-out logits for later ensemble search.

Reuses the E26 screen's EXACT eval slice + serialization contract so the output is
drop-in compatible with the existing pool cache (analysis/cache/e26_screen_logits*.npz):

  - slice = full_data seed-42 held-out ~3.5k:
      _, va = split_indices(labels, seed=42)
      _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    identical to screen_ensemble.py / finetune.py --full_data. The E32 models are full_data
    seed-42 → they HELD THIS SLICE OUT → clean, no leak.
  - serialize variant read per-run from ft_results*.csv (E32 = richargs); default richargs
    (NOT v1) since every E32 arm trained on richargs — a v1 mismatch would silently poison.
  - fp32 forward, length-agnostic batching, logits saved as {tag: (n,14) float32}.

Output default = analysis/cache/e26_screen_logits_s3.npz — matches the screen's shard glob
`e26_screen_logits_s*.npz`, so a future screen_ensemble.py run merges these automatically
and skips re-harvesting them. Incremental: re-run after A2/s2 sync to add those tags.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /home/kyusang/.conda/envs/dacon/bin/python experiments/ensemble/extract_e32_logits.py
"""
import argparse
import csv
import glob
import os

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

PREFIX = "ft_ibm-granite__granite-embedding-311m-multilingual-r2_"


def tag_of(run_dir):
    b = os.path.basename(run_dir.rstrip("/"))
    return b[len(PREFIX):] if b.startswith(PREFIX) else b


def load_serialize_map():
    """tag -> serialize variant, from every ft_results*.csv (last write wins)."""
    m = {}
    for p in glob.glob("output/pat/ft_results*.csv"):
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("tag") and row.get("serialize"):
                    m[row["tag"]] = row["serialize"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="output/pat/ft_*e32_*/",
                    help="run dirs to harvest (default all E32 arms present)")
    ap.add_argument("--out", default="analysis/cache/e26_screen_logits_s3.npz")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.model_selection import train_test_split

    samples, labels = load_samples(args.data_dir)
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    logger.info(f"held-out slice n={len(va_eval)} (full_data seed-42 held-out; E32 clean)")

    ser_map = load_serialize_map()
    # start from any existing shard so re-runs are incremental (A2/s2 added later)
    out = dict(np.load(args.out)) if os.path.exists(args.out) else {}
    texts_by_var = {}

    def get_texts(var):
        if var not in texts_by_var:
            texts_by_var[var] = build_texts([samples[i] for i in va_eval], variant=var)
        return texts_by_var[var]

    dirs = sorted(glob.glob(args.glob))
    logger.info(f"{len(dirs)} E32 run dir(s): {[tag_of(d) for d in dirs]}")
    for d in dirs:
        t = tag_of(d)
        if not os.path.exists(os.path.join(d, "model.safetensors")):
            logger.warning(f"skip {t}: no model.safetensors (still training / not synced?)")
            continue
        if t in out:
            logger.info(f"skip {t}: already in {args.out}")
            continue
        var = ser_map.get(t, "richargs")   # E32 = richargs; NEVER silent-v1
        logger.info(f"forward: {t}  (serialize={var})")
        tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            d, torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()
        txt = get_texts(var)
        chunks = []
        with torch.no_grad():
            for b in range(0, len(txt), args.batch_size):
                enc = tok(txt[b:b + args.batch_size], truncation=True,
                          max_length=args.max_len, padding=True,
                          return_tensors="pt").to("cuda")
                chunks.append(model(**enc).logits.float().cpu().numpy())
        out[t] = np.concatenate(chunks).astype(np.float32)
        del model
        torch.cuda.empty_cache()
        # sanity: argmax macro-F1 vs the recorded full_data CV
        from sklearn.metrics import f1_score
        pred = out[t].argmax(1)
        f1 = f1_score(y_ids[va_eval], pred,
                      labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)
        logger.success(f"  {t}: {out[t].shape}  self-check macro-F1={f1:.4f}")
        np.savez(args.out, **out)   # write after each model (crash-safe)

    logger.success(f"{len(out)} tag(s) in {args.out}: {sorted(out)}")


if __name__ == "__main__":
    main()
