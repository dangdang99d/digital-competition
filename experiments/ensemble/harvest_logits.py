"""Build experiments/ensemble/logits/ — one file per model, 3.5k held-out logits.

Goal: a single searchable directory of raw fp32 val logits on the SHARED 3.5k slice so
any ensemble combination can be scored offline without re-running GPU forwards.

Slice (byte-identical to finetune.py --full_data / screen_ensemble.py):
    _, va = split_indices(labels, seed=42)                       # 14k
    _, va_eval = train_test_split(va, 0.25, stratify, rs=42)     # 3.5k  (va_eval ⊂ va)
Any model trained on the seed-42 split (standard OR full_data) HELD OUT va_eval → clean.

Layout produced:
    logits/_meta.npz            va_idx (3500,), labels (3500,), classes (14,)
    logits/<tag>.npz            logits (3500,14) float32
    logits/manifest.jsonl       one row per candidate: decision + provenance + self-F1

Two stages:
  1. SPLIT existing 3.5k caches into per-model files (no GPU) — the "move" the user asked for.
  2. HARVEST every remaining loadable, non-leaking model (GPU), dropping self-F1 < 0.7.

Leakers (score HIGH because in-sample — F1 gate can't catch them, excluded by metadata):
  - e30_*_f{0..4}  : StratifiedGroupKFold OOF folds (full_data=False) — ~80% of va_eval in-train
  - e12_*_s43/s44  : seed 43/44 → different va → overlaps our seed-42 va_eval
  - e28 optuna trials (non-full_data): session-grouped folds → different split
Arch-modified loads (self-F1 ≈ 0.03, also skipped early to save GPU): a24_/b24_/e4_ffn/moe/smoke.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /home/kyusang/.conda/envs/dacon/bin/python experiments/ensemble/harvest_logits.py
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

OUTDIR = "experiments/ensemble/logits"
PREFIXES = ("ft_ibm-granite__granite-embedding-311m-multilingual-r2_",
            "ft_Qwen__Qwen3-Embedding-0.6B_", "ft_BAAI__bge-m3_",
            "ft_intfloat__multilingual-e5-small_")

# Existing 3.5k-slice caches to split into per-model files (all known-clean by construction).
EXISTING_TAGDICT = [   # {tag: (n,14)} npz
    "analysis/cache/e26_screen_logits.npz",
    "analysis/cache/e26_screen_logits_s0.npz",
    "analysis/cache/e26_screen_logits_s1.npz",
    "analysis/cache/e26_screen_logits_s2.npz",
    "analysis/cache/e26_screen_logits_s3.npz",
    "analysis/cache/e28fd_screen_logits.npz",
]
EXISTING_STACKED = glob.glob("output/moe_experts/val_logits_3500_*.npz")  # (K,n,14)+names+va_idx

# metadata-level leak exclusions (do NOT hold out seed-42 va_eval)
LEAK_RE = re.compile(r"(e30_(is3|aum06|e25c)_f\d|e12_granite_ls_s4[0-9])")
# won't load as a standard SeqCls / would be garbage — skip early (F1 gate would also drop)
INCOMPAT = ("a24_attn", "a24_sal", "b24_", "e4_ffn", "smoke", "ltp", "moe")
F1_FLOOR = 0.70


def tag_of(run_dir):
    b = os.path.basename(run_dir.rstrip("/"))
    for p in PREFIXES:
        if b.startswith(p):
            return b[len(p):]
    return b


def resolve_ckpt(run_dir):
    if os.path.exists(os.path.join(run_dir, "model.safetensors")):
        return run_dir
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    return None


def load_serialize_map():
    m = {}
    for p in glob.glob("output/pat/ft_results*.csv") + glob.glob("output/ft_results*.csv"):
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("tag") and row.get("serialize"):
                    m[row["tag"]] = row["serialize"]
    # E34 arms live outside ft_results*.csv — shared richargs anchor with E32
    for t in ("e34_anchor", "e34_a_fgm", "e34_b_pgd", "e34_c_awp"):
        m.setdefault(t, "richargs")
    return m


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--split_only", action="store_true", help="stage 1 only (no GPU)")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = os.path.join(OUTDIR, "manifest.jsonl")
    rows = []

    def record(**kw):
        rows.append(kw)
        logger.info(" | ".join(f"{k}={v}" for k, v in kw.items() if k != "note"))

    # ---- slice ----
    samples, labels = load_samples(args.data_dir)
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    ytrue = y_ids[va_eval]
    np.savez(os.path.join(OUTDIR, "_meta.npz"),
             va_idx=va_eval, labels=ytrue, classes=np.array(ALL_CLASSES))
    logger.success(f"_meta.npz written: n={len(va_eval)}")

    have = set()

    def save_logits(tag, arr, clean, serialize, source, note=""):
        arr = np.asarray(arr, np.float32)
        f1 = mf1(ytrue, arr.argmax(1))
        if f1 < F1_FLOOR:
            record(tag=tag, decision="drop_lowf1", self_f1=round(f1, 4),
                   serialize=serialize, source=source, note=note)
            return
        np.savez(os.path.join(OUTDIR, f"{tag}.npz"), logits=arr)
        have.add(tag)
        record(tag=tag, decision="ok", self_f1=round(f1, 4), clean=clean,
               serialize=serialize, source=source, note=note)

    # ---- stage 1: split existing caches (rows already in va_eval order — verified) ----
    for p in EXISTING_TAGDICT:
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        for t in d.files:
            a = d[t]
            if getattr(a, "shape", None) == (len(va_eval), len(ALL_CLASSES)) and t not in have:
                save_logits(t, a, clean=True, serialize="cache", source=os.path.basename(p))
    for p in EXISTING_STACKED:
        d = np.load(p, allow_pickle=True)
        assert np.array_equal(d["va_idx"], va_eval), f"{p} slice mismatch!"
        for i, nm in enumerate(d["names"]):
            t = str(nm)
            if t not in have:
                save_logits(t, d["logits"][i], clean=True, serialize="cache",
                            source=os.path.basename(p))
    logger.success(f"stage 1: {len(have)} per-model files from existing caches")

    if args.split_only:
        _flush(manifest, rows)
        return

    # ---- stage 2: harvest remaining loadable, non-leaking models ----
    ser_map = load_serialize_map()
    cand_dirs = []
    for pat in ("output/pat/ft_*/", "output/pat/*/", "output/e34/vast_r1/*/",
                "output/optuna/vast_r1/final_fd/ft_*/", "output/optuna/build_e28_trio/model/*/",
                "output/ft_*/"):
        cand_dirs += glob.glob(pat)
    seen_dir = set()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    texts_by_var = {}

    def get_texts(var):
        if var not in texts_by_var:
            texts_by_var[var] = build_texts([samples[i] for i in va_eval], variant=var)
        return texts_by_var[var]

    for d in sorted(set(cand_dirs)):
        ck = resolve_ckpt(d)
        if ck is None:
            continue
        t = tag_of(d)
        if t in seen_dir:
            continue
        seen_dir.add(t)
        if t in have:
            continue
        if LEAK_RE.search(t):
            record(tag=t, decision="skip_leak", source=d)
            continue
        if any(x in t for x in INCOMPAT):
            record(tag=t, decision="skip_incompat", source=d)
            continue
        var = ser_map.get(t)
        if var is None:
            record(tag=t, decision="skip_no_serialize", source=d,
                   note="no ft_results row; refuse to guess variant")
            continue
        try:
            tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                ck, torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()
            txt = get_texts(var)
            chunks = []
            with torch.no_grad():
                for b in range(0, len(txt), args.batch_size):
                    enc = tok(txt[b:b + args.batch_size], truncation=True,
                              max_length=args.max_len, padding=True,
                              return_tensors="pt").to("cuda")
                    chunks.append(model(**enc).logits.float().cpu().numpy())
            del model
            torch.cuda.empty_cache()
            save_logits(t, np.concatenate(chunks), clean=True, serialize=var, source=ck)
        except Exception as e:
            record(tag=t, decision="error", source=ck, note=str(e)[:200])
        _flush(manifest, rows)   # crash-safe

    _flush(manifest, rows)
    ok = [r for r in rows if r["decision"] == "ok"]
    logger.success(f"DONE: {len(ok)} models in {OUTDIR}/  ({len(rows)} candidates evaluated)")


def _flush(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
