"""E28 — harvest fp32 val logits of the four `--full_data` retrains on the shared 3.5k
held-out slice, in the SAME slice/format as experiments/ensemble/screen_ensemble.py so the
cache is mergeable with `e26_screen_logits.npz` for later combined E28+E26 ensemble screens.

Runs on the LOCAL GPU (RTX 4060 8GB) — bs 32 fp32 fits granite-311m@512.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u experiments/optuna/harvest_e28_logits.py
"""
import glob
import os

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.runlog import log_cmd

CACHE = "analysis/cache/e28fd_screen_logits.npz"
FINAL_FD = "output/optuna/vast_r1/final_fd"
BATCH, MAX_LEN = 32, 512


def main():
    log_cmd()
    os.makedirs("analysis/cache", exist_ok=True)

    # EXACT same 3.5k slice as screen_ensemble.py (string labels -> split_indices, then
    # the same stratified 25% held-out; all --full_data models held this out identically)
    samples, labels = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    ytrue = y_ids[va_eval]
    texts = build_texts([samples[i] for i in va_eval], variant="richargs")  # all E28 = richargs
    logger.info(f"held-out slice n={len(va_eval)}  (richargs)")

    cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    runs = sorted(glob.glob(f"{FINAL_FD}/ft_*full_t*"))
    assert runs, f"no E28 full_data runs under {FINAL_FD}"

    def mf1(p):
        return f1_score(ytrue, p, labels=np.arange(len(ALL_CLASSES)),
                        average="macro", zero_division=0)

    for d in runs:
        tag = "e28_" + d.split("_e28_")[-1]           # -> e28_full_t043
        if tag in cache:
            logger.info(f"cached: {tag}")
            continue
        logger.info(f"forward: {tag}")
        tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            d, local_files_only=True, torch_dtype=torch.float32).to("cuda").eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(texts), BATCH):
                enc = tok(texts[b:b + BATCH], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to("cuda")
                out.append(model(**enc).logits.float().cpu().numpy())
        cache[tag] = np.concatenate(out)
        del model
        torch.cuda.empty_cache()
        np.savez(CACHE, **cache)
        logger.success(f"{tag}: single-slice F1 = {mf1(cache[tag].argmax(1)):.4f}")

    logger.success(f"cache -> {CACHE} ({len([k for k in cache if k.startswith('e28_')])} E28 members)")
    print("\n## E28 full_data singles (3.5k held-out, raw)\n| member | F1 |\n|---|---|")
    for tag in sorted(k for k in cache if k.startswith("e28_")):
        print(f"| {tag} | {mf1(cache[tag].argmax(1)):.4f} |")


if __name__ == "__main__":
    main()
