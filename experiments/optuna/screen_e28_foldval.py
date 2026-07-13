"""E28 ensemble screen on the SESSION-FOLD-0 holdout (~14k, leak-free for the search models).

The optuna trials trained on session_fold_0 train and held out its ~14k val — so the banked
search checkpoints (t007/t009/t011/t017/t019/t023/t043) can be harvested on that holdout with
no leak, giving a 4× lower-noise ensemble screen than the 3.5k slice. Uniform softmax mean only
(NO weights, NO calibration — project invariant). Picks the combo; the DEPLOYMENT models are the
`--full_data` retrains; LB is the judge.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u experiments/optuna/screen_e28_foldval.py
"""
import glob
import itertools
import os

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import (ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples,
                      session_fold_indices)
from src.runlog import log_cmd

CACHE = "analysis/cache/e28_foldval_logits.npz"
TRIALS = "output/optuna/vast_r1/trials"
BATCH, MAX_LEN, FOLD = 32, 512, 0


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    log_cmd()
    os.makedirs("analysis/cache", exist_ok=True)

    samples, labels = load_samples("./data")          # labels = STRINGS (match finetune)
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = session_fold_indices(samples, labels, FOLD, n_splits=5, seed=42)
    ytrue = y_ids[va]
    texts = build_texts([samples[i] for i in va], variant="richargs")
    logger.info(f"session-fold-{FOLD} holdout n={len(va)} (leak-free for fold-{FOLD} search models)")

    def mf1(p):
        return f1_score(ytrue, p, labels=np.arange(len(ALL_CLASSES)),
                        average="macro", zero_division=0)

    cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    runs = sorted(glob.glob(f"{TRIALS}/ft_*e28_t0*"))
    runs = [d for d in runs if os.path.isfile(f"{d}/model.safetensors")]
    assert runs, f"no search checkpoints under {TRIALS}"

    for d in runs:
        tag = "t" + d.split("_e28_t")[-1]             # -> t043
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
        logger.success(f"{tag}: single F1 = {mf1(cache[tag].argmax(1)):.4f}")

    # ---- ensemble screen (uniform softmax mean) ----
    tags = sorted(cache.keys())
    P = {t: softmax(cache[t]) for t in tags}
    singles = sorted(((mf1(P[t].argmax(1)), t) for t in tags), reverse=True)

    def ens(members):
        return mf1(np.mean([P[t] for t in members], 0).argmax(1))

    print("\n## Singles (session-fold-0 holdout, raw)\n| model | F1 |\n|---|---|")
    for f, t in singles:
        print(f"| {t} | {f:.4f} |")

    print("\n## Best pairs\n| pair | F1 |\n|---|---|")
    pairs = sorted(((ens(c), c) for c in itertools.combinations(tags, 2)), reverse=True)
    for f, c in pairs[:6]:
        print(f"| {'+'.join(c)} | {f:.4f} |")

    print("\n## Best trios\n| trio | F1 |\n|---|---|")
    trios = sorted(((ens(c), c) for c in itertools.combinations(tags, 3)), reverse=True)
    for f, c in trios[:6]:
        print(f"| {'+'.join(c)} | {f:.4f} |")

    print("\n## Best quads\n| quad | F1 |\n|---|---|")
    quads = sorted(((ens(c), c) for c in itertools.combinations(tags, 4)), reverse=True)
    for f, c in quads[:4]:
        print(f"| {'+'.join(c)} | {f:.4f} |")

    best_single = singles[0]
    logger.success(f"best single {best_single[1]} {best_single[0]:.4f} | "
                   f"best pair {pairs[0][1]} {pairs[0][0]:.4f} | "
                   f"best trio {trios[0][1]} {trios[0][0]:.4f}")


if __name__ == "__main__":
    main()
