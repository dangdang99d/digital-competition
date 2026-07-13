"""Recovery pass: harvest the high-value models that harvest_logits.py skipped for
`skip_no_serialize` (dir-tag had no ft_results row). Each entry below carries an
EXPLICIT, human-verified serialize variant so nothing is guessed. Same 3.5k slice,
same F1>=0.70 gate, appends to logits/manifest.jsonl.
"""
import json
import os

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

OUTDIR = "experiments/ensemble/logits"
F1_FLOOR = 0.70

# tag -> (checkpoint_dir, serialize)   — all richargs-era full_data seed-42 (hold out va_eval)
# NOTE: qwen3_champion_pruned is EXCLUDED — pruned-vocab model needs the remap.npy input path
# (full tokenizer -> out-of-range IDs -> CUDA assert with plain from_pretrained); its unpruned
# sources e8b_qwen3_richargs_full / e8b_ls_qwen3_richargs_full are already in the pool.
RECOVER = {
    "soup_granite_ls":       ("output/pat/soup_granite_ls", "richargs"),
    "e28_full_t019":         ("output/optuna/vast_r1/final_fd/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e28_full_t019", "richargs"),
    "e26s_ba_a70_T2":        ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e26s_ba_a70_T2", "richargs"),
    "e26s_t1_a70_T2":        ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e26s_t1_a70_T2", "richargs"),
    "e26s_t1_soft":          ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e26s_t1_soft", "richargs"),
    "e26s_t2_a70_T2":        ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e26s_t2_a70_T2", "richargs"),
}


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def main():
    from sklearn.model_selection import train_test_split
    samples, labels = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    meta = np.load(os.path.join(OUTDIR, "_meta.npz"))
    assert np.array_equal(meta["va_idx"], va_eval), "slice drift vs _meta.npz!"
    ytrue = y_ids[va_eval]

    rows = []
    for tag, (d, var) in RECOVER.items():
        if os.path.exists(os.path.join(OUTDIR, f"{tag}.npz")):
            logger.info(f"{tag}: already present, skip")
            continue
        try:
            tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                d, torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()
            txt = build_texts([samples[i] for i in va_eval], variant=var)
            chunks = []
            with torch.no_grad():
                for b in range(0, len(txt), 32):
                    enc = tok(txt[b:b + 32], truncation=True, max_length=512,
                              padding=True, return_tensors="pt").to("cuda")
                    chunks.append(model(**enc).logits.float().cpu().numpy())
            del model
            torch.cuda.empty_cache()
            arr = np.concatenate(chunks).astype(np.float32)
            f1 = mf1(ytrue, arr.argmax(1))
            if f1 < F1_FLOOR:
                rows.append({"tag": tag, "decision": "drop_lowf1", "self_f1": round(f1, 4),
                             "serialize": var, "source": d})
                logger.warning(f"{tag}: F1={f1:.4f} < {F1_FLOOR} — dropped")
                continue
            np.savez(os.path.join(OUTDIR, f"{tag}.npz"), logits=arr)
            rows.append({"tag": tag, "decision": "ok", "self_f1": round(f1, 4),
                         "clean": True, "serialize": var, "source": d, "note": "recovered"})
            logger.success(f"{tag}: {arr.shape} F1={f1:.4f} -> saved")
        except Exception as e:
            rows.append({"tag": tag, "decision": "error", "source": d, "note": str(e)[:200]})
            logger.error(f"{tag}: {e}")

    with open(os.path.join(OUTDIR, "manifest.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r["decision"] == "ok"]
    logger.success(f"recovery: {len(ok)}/{len(RECOVER)} added")


if __name__ == "__main__":
    main()
