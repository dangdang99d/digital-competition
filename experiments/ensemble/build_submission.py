"""E26 Path A — build the two-granite ensemble submission zip.

Both members are pruned with the SAME keep-set (used richargs tokens ∪ specials ∪
[0, margin)) so one tokenizer + one remap.npy serves the pair — the memory-noted reuse.
Each member: slice embedding rows -> fp16 -> model/member_<i>_<tag>/. Then script.py
(= script_ensemble.py), requirements.txt, PARITY GATE per member (pruned-fp16 pipeline
argmax vs the member's cached fp32 full-vocab screen logits on the 3.5k slice), zip.

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. nohup /home/ocean/miniconda3/envs/dacon/bin/python -u \
    experiments/ensemble/build_submission.py \
    --runs output/pat/ft_..._e8a_ls_richargs_full,output/pat/ft_..._<second> \
    --tags e8a_ls,<tag2> --out submissions/submit_0709_ensemble2.zip \
    > sbatch/logs/e26_build.log 2>&1 &

Aborts if: parity mismatch > 0.1% argmax flips, or zip > 950MB.
"""
import argparse
import glob
import os
import shutil
import zipfile

import numpy as np
import torch
from loguru import logger

from src.data import SERIALIZE_VARIANTS, CLASS_TO_ID, build_texts, load_samples, split_indices
from src.prune_vocab import collect_used_ids

SCRIPT_TEMPLATE = "experiments/ensemble/script_ensemble.py"
SCREEN_CACHE = "analysis/cache/e26_screen_logits.npz"
REQUIREMENTS = "torch\ntransformers==4.51.3\nnumpy\nscikit-learn\n"


def resolve_ckpt(run_dir):
    if os.path.exists(os.path.join(run_dir, "model.safetensors")):
        return run_dir
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    raise FileNotFoundError(run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma: 2+ granite run dirs (richargs-trained)")
    ap.add_argument("--tags", required=True, help="comma: short names, same order")
    ap.add_argument("--out", required=True, help="output zip path")
    ap.add_argument("--margin", type=int, default=50000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--parity_n", type=int, default=3000)
    ap.add_argument("--parity_tol", type=float, default=0.001)
    ap.add_argument("--build_dir", default="build_e26")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    runs = args.runs.split(",")
    tags = args.tags.split(",")
    assert len(runs) == len(tags) >= 2
    dev = "cuda"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # ---- shared keep-set + remap (tokenizer-level: model-independent) ----
    ck0 = resolve_ckpt(runs[0])
    tok = AutoTokenizer.from_pretrained(ck0, local_files_only=True)
    used = collect_used_ids(tok, "./data/train.jsonl", args.max_len,
                            SERIALIZE_VARIANTS["richargs"])
    keep = sorted(used | set(tok.all_special_ids) | set(range(args.margin)))
    remap = np.full(tok.vocab_size if tok.vocab_size else max(keep) + 1, -1, dtype=np.int64)
    # (granite vocab_size from config below; build remap over the model's emb rows)
    logger.info(f"keep {len(keep):,} tokens (used {len(used):,} + margin {args.margin:,})")

    root = os.path.join(args.build_dir, "model")
    if os.path.exists(args.build_dir):
        shutil.rmtree(args.build_dir)
    os.makedirs(root)
    tok.save_pretrained(os.path.join(root, "tokenizer"))

    # ---- prune + fp16 each member with the SAME keep ----
    for i, (run, tag) in enumerate(zip(runs, tags)):
        ck = resolve_ckpt(run)
        model = AutoModelForSequenceClassification.from_pretrained(
            ck, local_files_only=True, torch_dtype=torch.float32)
        emb = model.get_input_embeddings()
        if remap[0] == -1 and (remap == -1).all():           # build once, sized to emb
            remap = np.full(emb.num_embeddings, -1, dtype=np.int64)
            for new, old in enumerate(keep):
                remap[old] = new
            fallback = tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id
            remap[remap == -1] = int(remap[fallback])
            np.save(os.path.join(root, "remap.npy"), remap.astype(np.int32))
        new_emb = torch.nn.Embedding(len(keep), emb.embedding_dim)
        new_emb.weight.data = emb.weight.data[keep].clone()
        model.set_input_embeddings(new_emb)
        model.config.vocab_size = len(keep)
        if model.config.pad_token_id is not None:
            model.config.pad_token_id = int(remap[model.config.pad_token_id])
        model.half()
        mdir = os.path.join(root, f"member_{i}_{tag}")
        model.save_pretrained(mdir)
        sz = sum(os.path.getsize(os.path.join(mdir, f)) for f in os.listdir(mdir))
        logger.info(f"member_{i}_{tag}: pruned fp16 -> {mdir} ({sz/1e6:.0f} MB)")
        del model

    shutil.copy(SCRIPT_TEMPLATE, os.path.join(args.build_dir, "script.py"))
    with open(os.path.join(args.build_dir, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS)

    # ---- parity gate: shipped pipeline vs cached fp32 full-vocab screen logits ----
    samples, labels = load_samples("./data")
    y = np.array([CLASS_TO_ID[a] for a in labels])
    from sklearn.model_selection import train_test_split
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y[va], random_state=42)
    rows = va_eval[: args.parity_n]
    texts = build_texts([samples[i] for i in rows], variant="richargs")
    cache = dict(np.load(SCREEN_CACHE))
    for p in sorted(glob.glob(SCREEN_CACHE.replace(".npz", "_s*.npz"))):
        cache.update(dict(np.load(p)))
    rt = torch.from_numpy(remap).long().to(dev)
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tok)
    enc = tok(texts, truncation=True, max_length=args.max_len)
    enc = [{"input_ids": enc["input_ids"][k], "attention_mask": enc["attention_mask"][k]}
           for k in range(len(texts))]
    for i, tag in enumerate(tags):
        assert tag in cache, f"{tag} missing from screen cache — run the screen first"
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(root, f"member_{i}_{tag}"), local_files_only=True,
            torch_dtype=torch.float16).to(dev).eval()
        preds = []
        with torch.no_grad():
            for b in range(0, len(enc), 128):
                batch = {k: v.to(dev) for k, v in collator(enc[b:b + 128]).items()}
                batch["input_ids"] = rt[batch["input_ids"]]
                preds.append(model(**batch).logits.float().argmax(-1).cpu().numpy())
        preds = np.concatenate(preds)
        ref = cache[tag][: len(rows)].argmax(1)
        flips = float((preds != ref).mean())
        logger.info(f"parity {tag}: argmax flips {flips:.4%} (tol {args.parity_tol:.2%})")
        assert flips <= args.parity_tol, f"PARITY FAIL {tag}: {flips:.4%}"
        del model
        torch.cuda.empty_cache()

    # ---- zip ----
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, files in os.walk(args.build_dir):
            for f in files:
                full = os.path.join(dirpath, f)
                z.write(full, os.path.relpath(full, args.build_dir))
    sz = os.path.getsize(args.out) / 1e6
    logger.success(f"{args.out}: {sz:.0f} MB {'✓ under cap' if sz < 950 else '✗ OVER 950MB — ABORT'}")
    assert sz < 950


if __name__ == "__main__":
    main()
