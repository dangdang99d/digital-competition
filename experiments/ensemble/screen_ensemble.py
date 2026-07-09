"""E26 Phase 0 — ensemble screen over existing checkpoints on the shared 3.5k held-out.

Harvests raw fp32 val logits for every eligible run under output/pat/ (serialization variant
read from that run's ft_results*.csv row), caches them, then evaluates ensembles by UNIFORM
softmax mean (no weights, no calibration — project invariant):
  - every single model (sanity vs known numbers)
  - all pairs (top-15 by macro-F1)
  - greedy forward selection (Caruana, uniform, no replacement) with the full trajectory
  - pairwise disagreement matrix for the top members (diversity read)

Eval slice = full_data seed-42 held-out (~3.5k): train_test_split(va, test_size=0.25,
stratify, random_state=42) — identical to finetune.py --full_data. Standard-split models never
saw any of the 14k val, full_data models held this slice out → fair for all.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ocean/dacon nohup \
    /home/ocean/miniconda3/envs/dacon/bin/python -u experiments/ensemble/screen_ensemble.py \
    > sbatch/logs/e26_screen.log 2>&1 &
"""
import argparse
import glob
import os

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

CACHE = "analysis/cache/e26_screen_logits.npz"
# substrings that disqualify a run dir (specialists, smoke, reduced-input pipelines)
EXCLUDE = ("ceil_", "smoke", "a24_attn", "a24_sal", "b24_", "ltp")


def resolve_ckpt(run_dir):
    if os.path.exists(os.path.join(run_dir, "model.safetensors")):
        return run_dir
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    return None


def tag_of(run_dir):
    b = os.path.basename(run_dir.rstrip("/"))
    for pre in ("ft_ibm-granite__granite-embedding-311m-multilingual-r2_",
                "ft_Qwen__Qwen3-Embedding-0.6B_", "ft_BAAI__bge-m3_"):
        if b.startswith(pre):
            return b[len(pre):]
    return b


def load_serialize_map():
    """tag -> serialize variant, from every ft_results*.csv (last write wins)."""
    import csv
    m = {}
    for p in glob.glob("output/pat/ft_results*.csv"):
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("tag") and row.get("serialize"):
                    m[row["tag"]] = row["serialize"]
    return m


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N held-out rows")
    ap.add_argument("--greedy_len", type=int, default=6)
    args = ap.parse_args()

    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    def mf1(y, p):
        return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro",
                        zero_division=0)

    samples, labels = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    if args.limit:
        va_eval = va_eval[: args.limit]
    ytrue = y_ids[va_eval]
    logger.info(f"held-out slice n={len(va_eval)}")

    ser_map = load_serialize_map()
    runs = []
    for d in sorted(glob.glob("output/pat/ft_*/")):
        t = tag_of(d)
        if any(x in t for x in EXCLUDE) or resolve_ckpt(d) is None:
            continue
        # a24_anchor is richargs full-input trained via reduced_ids of the FULL input — plain richargs at eval
        var = ser_map.get(t, "v1")
        runs.append((t, d, var))
    logger.info(f"pool: {len(runs)} runs " + str([(t, v) for t, d, v in runs]))

    texts_by_var = {}
    cache = dict(np.load(CACHE)) if os.path.exists(CACHE) and not args.limit else {}

    def get_texts(var):
        if var not in texts_by_var:
            texts_by_var[var] = build_texts([samples[i] for i in va_eval], variant=var)
        return texts_by_var[var]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for t, d, var in runs:
        if t in cache:
            continue
        logger.info(f"forward: {t} ({var})")
        try:
            tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                resolve_ckpt(d), torch_dtype=torch.float32,
                trust_remote_code=True).to("cuda").eval()
            txt = get_texts(var)
            out = []
            with torch.no_grad():
                for b in range(0, len(txt), args.batch_size):
                    enc = tok(txt[b:b + args.batch_size], truncation=True,
                              max_length=args.max_len, padding=True,
                              return_tensors="pt").to("cuda")
                    out.append(model(**enc).logits.float().cpu().numpy())
            cache[t] = np.concatenate(out)
            del model
            torch.cuda.empty_cache()
            if not args.limit:
                np.savez(CACHE, **cache)
        except Exception as e:
            logger.error(f"{t}: skipped — {e}")

    # ---- ensemble math (uniform softmax mean) ----
    tags = [t for t, _, _ in runs if t in cache]
    P = {t: softmax(cache[t]) for t in tags}
    singles = sorted(((mf1(ytrue, P[t].argmax(1)), t) for t in tags), reverse=True)
    print("\n## Singles (raw, 3.5k held-out)\n")
    print("| model | F1 |\n|---|---|")
    for f, t in singles:
        print(f"| {t} | {f:.4f} |")

    def ens_f1(members):
        m = sum(P[t] for t in members) / len(members)
        return mf1(ytrue, m.argmax(1))

    print("\n## Top pairs\n")
    pairs = []
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            pairs.append((ens_f1([a, b]), a, b))
    pairs.sort(reverse=True)
    print("| pair | F1 | vs best single |\n|---|---|---|")
    best_single = singles[0][0]
    for f, a, b in pairs[:15]:
        print(f"| {a} + {b} | {f:.4f} | {f - best_single:+.4f} |")

    print("\n## Greedy (Caruana, uniform, no replacement)\n")
    members, cur = [], 0.0
    pool = set(tags)
    print("| step | added | ensemble F1 | Δ |\n|---|---|---|---|")
    for step in range(args.greedy_len):
        f_best, t_best = max((ens_f1(members + [t]), t) for t in pool)
        if members and f_best <= cur:
            print(f"| {step + 1} | (no gain — stop) | {cur:.4f} | |")
            break
        members.append(t_best)
        pool.discard(t_best)
        print(f"| {step + 1} | {t_best} | {f_best:.4f} | {f_best - cur:+.4f} |")
        cur = f_best

    # ---- diversity: pairwise disagreement among the top-8 singles ----
    top8 = [t for _, t in singles[:8]]
    print("\n## Disagreement (fraction of rows with different argmax), top-8 singles\n")
    print("| |" + "|".join(t[:14] for t in top8) + "|")
    print("|---|" + "---|" * len(top8))
    preds = {t: P[t].argmax(1) for t in top8}
    for a in top8:
        row = [f"{(preds[a] != preds[b]).mean():.3f}" if a != b else "—" for b in top8]
        print(f"| {a[:14]} |" + "|".join(row) + "|")


if __name__ == "__main__":
    main()
