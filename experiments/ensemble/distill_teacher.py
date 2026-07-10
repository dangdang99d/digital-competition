"""E26 Path B (E10 revival) — build the ensemble TEACHER for distillation to one granite.

1. Forward each member over ALL 70k train rows (each with ITS OWN serialization, fp32),
   cached per member -> analysis/cache/e26_train_logits_<tag>.npz (idempotent).
2. Teacher = uniform mean of member softmaxes; saved as log-probabilities in the "logits"
   key that finetune.py --distill_from expects (softmax(log p) == p, and --distill_T
   tempers them exactly like real logits).

Members MAY include qwen3_ls — unpackageable for submission (9:18) but a legal teacher
(distillation is offline). Student = FROM-SCRATCH granite champion recipe (E24 lesson:
never warm-start) with CE+KD (--loss ls is not combinable with --distill_from; KD soft
targets act as learned label smoothing).

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. nohup /home/ocean/miniconda3/envs/dacon/bin/python -u \
    experiments/ensemble/distill_teacher.py --members e8a_ls_richargs_full,... \
    > sbatch/logs/e26_teacher.log 2>&1 &
"""
import argparse
import csv
import glob
import os

import numpy as np
from loguru import logger

from src.data import build_texts, load_samples

CACHE_FMT = "analysis/cache/e26_train_logits_{}.npz"


def resolve_ckpt(run_dir):
    if os.path.exists(os.path.join(run_dir, "model.safetensors")):
        return run_dir
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    return None


def find_run(tag):
    hits = [d for d in glob.glob("output/pat/ft_*/")
            if os.path.basename(d.rstrip("/")).endswith(tag)]
    assert len(hits) == 1, f"{tag}: expected exactly 1 run dir, got {hits}"
    return hits[0]


def serialize_of(tag):
    m = {}
    for p in glob.glob("output/pat/ft_results*.csv"):
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("tag") and row.get("serialize"):
                    m[row["tag"]] = row["serialize"]
    return m.get(tag, "richargs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma: run TAGS (screen winners)")
    ap.add_argument("--out", default="analysis/cache/e26_teacher.npz")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    samples, _ = load_samples("./data")
    tags = args.members.split(",")
    texts_by_var = {}

    for tag in tags:
        cpath = CACHE_FMT.format(tag)
        if os.path.exists(cpath):
            logger.info(f"{tag}: cached")
            continue
        run = find_run(tag)
        var = serialize_of(tag)
        if var not in texts_by_var:
            texts_by_var[var] = build_texts(samples, variant=var)
        txt = texts_by_var[var]
        logger.info(f"{tag} ({var}): forward over {len(txt)} rows")
        tok = AutoTokenizer.from_pretrained(run, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            resolve_ckpt(run), torch_dtype=torch.float32,
            trust_remote_code=True).to("cuda").eval()
        order = sorted(range(len(txt)), key=lambda i: len(txt[i]))   # cheap length-sort
        out = np.zeros((len(txt), model.config.num_labels), dtype=np.float32)
        with torch.no_grad():
            for b in range(0, len(order), args.batch_size):
                idx = order[b:b + args.batch_size]
                enc = tok([txt[i] for i in idx], truncation=True, max_length=args.max_len,
                          padding=True, return_tensors="pt").to("cuda")
                out[idx] = model(**enc).logits.float().cpu().numpy()
        np.savez(cpath, logits=out)
        logger.success(f"{tag} -> {cpath}")
        del model
        torch.cuda.empty_cache()

    # ---- teacher = uniform mean of member softmaxes, stored as log-probs ----
    def sm(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    mean_p = np.mean([sm(np.load(CACHE_FMT.format(t))["logits"]) for t in tags], axis=0)
    teacher = np.log(np.clip(mean_p, 1e-9, 1.0)).astype(np.float32)
    np.savez(args.out, logits=teacher, members=np.array(tags))
    logger.success(f"teacher ({len(tags)} members) -> {args.out}")
    print(f"""
STUDENT (from-scratch champion recipe + KD; sweep alpha/T if the first read is close):
CUDA_VISIBLE_DEVICES=<g> nohup /home/ocean/miniconda3/envs/dacon/bin/python -u -m src.finetune \\
  --distill_from {args.out} --distill_alpha 0.7 --distill_T 3.0 \\
  --tag e26_distill_a70_T3 \\
  --model ibm-granite/granite-embedding-311m-multilingual-r2 --serialize richargs --full_data \\
  --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4 --max_len 512 --seed 42 \\
  --out_dir ./output/pat --results_name ft_results_e26.csv \\
  > sbatch/logs/e26_distill_a70_T3.log 2>&1 &""")


if __name__ == "__main__":
    main()
