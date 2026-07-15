"""E38 honest ensemble screen on the session-grouped fold-0 14k holdout.

The 4 fold models (t070/t031/t040/t001) were each trained on fold-0's 56k train and
held out the SAME 14k session-grouped holdout -> this holdout is leak-free for ALL of
them, so single-model AND ensemble F1 measured here are honest (unlike the 3.5k slice,
which those models may have trained on).

Per model: richargs serialize -> tokenize -> softmax probs on the 14k holdout, cached.
Then every 1/2/3/4-model uniform-prob-mean combo scored by macro-F1.
Single-model F1 must match the training-reported fold F1 (~0.7806 etc) -> proves the
split byte-matches finetune.py's --session_fold 0.
"""
import itertools
import os

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score

from src.data import (ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples,
                      session_fold_indices)

MODELS = {
    "t070": "output/e38/vast_search/fold_models/t070_fold56k",
    "t031": "output/e38/vast_search/fold_models/t031_fold56k",
    "t040": "output/e38/vast_search/fold_models/t040_fold56k",
    "t001": "output/e38/vast_search/fold_models/t001_fold56k",
}
CACHE = "output/e38/fold_screen_probs.npz"
MAX_LEN = 512
BS = 32


def mf1(y, p):
    return f1_score(y, p, labels=list(range(len(ALL_CLASSES))), average="macro",
                    zero_division=0)


def main():
    samples, y = load_samples("./data")
    texts = build_texts(samples, variant="richargs")
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = session_fold_indices(samples, y, 0, n_splits=5, seed=42)
    logger.info(f"fold-0 holdout: {len(va)} rows (train {len(tr)})")
    va_texts = [texts[i] for i in va]
    y_va = y_ids[va]

    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        probs = {k: d[k] for k in MODELS}
        assert np.array_equal(d["y_va"], y_va), "cache holdout mismatch — delete cache"
        logger.info("loaded cached probs")
    else:
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)
        probs = {}
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for tag, path in MODELS.items():
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
            out = np.zeros((len(va_texts), len(ALL_CLASSES)), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, len(va_texts), BS):
                    batch = va_texts[i:i + BS]
                    enc = tok(batch, truncation=True, max_length=MAX_LEN,
                              padding=True, return_tensors="pt").to(dev)
                    logit = model(**enc).logits.float()
                    out[i:i + BS] = torch.softmax(logit, -1).cpu().numpy()
                    if i % (BS * 50) == 0:
                        logger.info(f"{tag}: {i}/{len(va_texts)}")
            probs[tag] = out
            f1 = mf1(y_va, out.argmax(1))
            logger.info(f"{tag} single-model fold-0 F1 = {f1:.4f}")
            del model
            torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, y_va=y_va, **probs)
        logger.info(f"cached -> {CACHE}")

    # ---- score every combo (uniform prob mean) ----
    tags = list(MODELS)
    print("\n=== single ===")
    singles = {t: mf1(y_va, probs[t].argmax(1)) for t in tags}
    for t, f in sorted(singles.items(), key=lambda x: -x[1]):
        print(f"  {t:6s} {f:.4f}")

    results = []
    for r in (2, 3, 4):
        for combo in itertools.combinations(tags, r):
            mean = np.mean([probs[t] for t in combo], axis=0)
            results.append(("+".join(combo), mf1(y_va, mean.argmax(1))))
    print("\n=== ensembles (uniform prob mean) ===")
    for name, f in sorted(results, key=lambda x: -x[1]):
        print(f"  {name:24s} {f:.4f}")

    best_single = max(singles.items(), key=lambda x: x[1])
    best_ens = max(results, key=lambda x: x[1])
    print(f"\nBEST single : {best_single[0]} {best_single[1]:.4f}")
    print(f"BEST ensemble: {best_ens[0]} {best_ens[1]:.4f}  "
          f"(+{best_ens[1]-best_single[1]:.4f} over best single)")


if __name__ == "__main__":
    main()
