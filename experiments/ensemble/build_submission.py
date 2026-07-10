"""E26 Path A — build a two-granite ensemble submission zip.

Members may use DIFFERENT serializations (richargs/richmeta): the keep-set is the UNION of
used token ids across the zip's variants, so one tokenizer + one remap.npy serves all
members; each member dir carries serialize_variant.json for script.py. Members are pruned
with that shared keep-set, cast fp16, saved to model/member_<i>_<tag>/.

Parity gate (pruned-fp16 pipeline argmax vs the member's cached fp32 full-vocab screen
logits on the 3.5k slice) needs a GPU — with --skip_parity the zip is built WITHOUT it and
MUST NOT be uploaded until `--parity_only` passes (both modes are idempotent on build_dir).

  PYTHONPATH=. nohup /home/ocean/miniconda3/envs/dacon/bin/python -u \
    experiments/ensemble/build_submission.py \
    --runs <run_dir_a>,<run_dir_b> --tags <a>,<b> --variants richargs,richargs \
    --out submissions/submit_0710_ens2.zip --build_dir <local>/build_e26 --skip_parity \
    > sbatch/logs/e26_build.log 2>&1 &

Aborts if: parity mismatch > 0.1% argmax flips (when run), or zip > 950MB.
"""
import argparse
import glob
import json
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


def parity(args, tags, root, tok, remap, dev="cuda"):
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
    from sklearn.model_selection import train_test_split
    samples, labels = load_samples("./data")
    y = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y[va], random_state=42)
    rows = va_eval[: args.parity_n]
    cache = {}
    for p in [SCREEN_CACHE] + sorted(glob.glob(SCREEN_CACHE.replace(".npz", "_s*.npz"))):
        if os.path.exists(p):
            d = np.load(p)
            cache.update({k: d[k] for k in d.files})
    rt = torch.from_numpy(remap).long().to(dev)
    collator = DataCollatorWithPadding(tok)
    variants = args.variants.split(",")
    for i, (tag, var) in enumerate(zip(tags, variants)):
        assert tag in cache, f"{tag} missing from screen cache — run the screen first"
        texts = build_texts([samples[j] for j in rows], variant=var)
        enc = tok(texts, truncation=True, max_length=args.max_len)
        enc = [{"input_ids": enc["input_ids"][k], "attention_mask": enc["attention_mask"][k]}
               for k in range(len(texts))]
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
        logger.info(f"parity {tag} ({var}): argmax flips {flips:.4%} (tol {args.parity_tol:.2%})")
        assert flips <= args.parity_tol, f"PARITY FAIL {tag}: {flips:.4%}"
        del model
        torch.cuda.empty_cache()
    logger.success("parity PASS for all members")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma: 2+ granite run dirs")
    ap.add_argument("--tags", required=True, help="comma: short names, same order")
    ap.add_argument("--variants", required=True,
                    help="comma: serialization per member (richargs|richmeta), same order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=int, default=50000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--parity_n", type=int, default=3000)
    ap.add_argument("--parity_tol", type=float, default=0.001)
    ap.add_argument("--build_dir", default="build_e26")
    ap.add_argument("--skip_parity", action="store_true",
                    help="build on CPU without the GPU parity gate — zip NOT uploadable "
                         "until a later --parity_only run passes")
    ap.add_argument("--parity_only", action="store_true",
                    help="run ONLY the parity gate against an existing build_dir")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    runs, tags = args.runs.split(","), args.tags.split(",")
    variants = args.variants.split(",")
    assert len(runs) == len(tags) == len(variants) >= 2
    assert all(v in ("richargs", "richmeta") for v in variants)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    root = os.path.join(args.build_dir, "model")

    if args.parity_only:
        tok = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"),
                                            local_files_only=True)
        remap = np.load(os.path.join(root, "remap.npy")).astype(np.int64)
        parity(args, tags, root, tok, remap)
        return

    ck0 = resolve_ckpt(runs[0])
    tok = AutoTokenizer.from_pretrained(ck0, local_files_only=True)
    used = set()
    for v in sorted(set(variants)):
        u = collect_used_ids(tok, "./data/train.jsonl", args.max_len, SERIALIZE_VARIANTS[v])
        logger.info(f"used ids ({v}): {len(u):,}")
        used |= u
    keep = sorted(used | set(tok.all_special_ids) | set(range(args.margin)))
    logger.info(f"keep {len(keep):,} tokens (union-used {len(used):,} + margin {args.margin:,})")

    if os.path.exists(args.build_dir):
        shutil.rmtree(args.build_dir)
    os.makedirs(root)
    tok.save_pretrained(os.path.join(root, "tokenizer"))

    remap = None
    for i, (run, tag, var) in enumerate(zip(runs, tags, variants)):
        ck = resolve_ckpt(run)
        model = AutoModelForSequenceClassification.from_pretrained(
            ck, local_files_only=True, torch_dtype=torch.float32)
        emb = model.get_input_embeddings()
        if remap is None:
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
        json.dump({"variant": var}, open(os.path.join(mdir, "serialize_variant.json"), "w"))
        sz = sum(os.path.getsize(os.path.join(mdir, f)) for f in os.listdir(mdir))
        logger.info(f"member_{i}_{tag} ({var}): pruned fp16 -> {mdir} ({sz/1e6:.0f} MB)")
        del model

    shutil.copy(SCRIPT_TEMPLATE, os.path.join(args.build_dir, "script.py"))
    with open(os.path.join(args.build_dir, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS)

    if args.skip_parity:
        logger.warning("PARITY SKIPPED (no GPU) — do NOT upload before a --parity_only PASS")
    else:
        parity(args, tags, root, tok, remap)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, files in os.walk(args.build_dir):
            for f in files:
                full = os.path.join(dirpath, f)
                z.write(full, os.path.relpath(full, args.build_dir))
    sz = os.path.getsize(args.out) / 1e6
    logger.success(f"{args.out}: {sz:.0f} MB {'OK under cap' if sz < 950 else 'OVER 950MB'}")
    assert sz < 950


if __name__ == "__main__":
    main()
