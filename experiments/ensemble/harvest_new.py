"""Harvest 3.5k-slice logits for the NEW experiments (E35 loss sweep, E38 AWP-optuna, SWA).
Same slice + F1>=0.70 gate as the pool. Serialize via variant-argmax (try v1/richargs/richmeta,
keep the best-scoring — the true training format dominates, so no guessing). Skips already-present
tags; appends to logits/manifest.jsonl. EXCLUDES E38 fold_models (optuna search folds → leak).
"""
import glob
import json
import os

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

OUTDIR = "experiments/ensemble/logits"
F1_FLOOR = 0.70
VARIANTS = ["richargs", "v1", "richmeta"]
E38P = "output/e38/promote_top4/ft_ibm-granite__granite-embedding-311m-multilingual-r2_"
E35 = "output/e35"

# tag -> checkpoint dir  (all seed-42 standard/full_data granite classifiers -> hold out va_eval)
NEW = {
    "e38_t001fd": E38P + "e38_t001fd",
    "e38_t031fd": E38P + "e38_t031fd",
    "e38_t040fd": E38P + "e38_t040fd",
    "e38_t070fd": E38P + "e38_t070fd",
    "e38_t031_elr": "output/e38/elr_awp/e38_t031_elr",
    "e38_t040_elr": "output/e38/elr_awp/e38_t040_elr",
    "e38_t070_elr": "output/e38/elr_awp/e38_t070_elr",
    "e35_ce": f"{E35}/ce/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e35_ce",
    "e35_ls": f"{E35}/ls/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e35_ls",
    "e35_gce": f"{E35}/gce/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e35_gce",
    "e35_sce": f"{E35}/sce/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e35_sce",
    "e35_apl": f"{E35}/apl/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e35_apl",
    "e35_boot": f"{E35}/boot/ft_ibm-granite__granite-embedding-311m-multilingual-r2_e35_boot",
    "e35_fd_elr": f"{E35}/fd_elr_model",
    "e35_fd_ls": f"{E35}/fd_ls_model",
    "g_awp_swa": "output/vast_44742238/ft_ibm-granite__granite-embedding-311m-multilingual-r2_g_awp_swa",
}


def resolve_ckpt(d):
    if os.path.exists(os.path.join(d, "model.safetensors")):
        return d
    cks = sorted(glob.glob(os.path.join(d, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    return None


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
    txt = {v: build_texts([samples[i] for i in va_eval], variant=v) for v in VARIANTS}

    rows = []
    for tag, d in NEW.items():
        if os.path.exists(os.path.join(OUTDIR, f"{tag}.npz")):
            logger.info(f"{tag}: present, skip"); continue
        ck = resolve_ckpt(d)
        if ck is None:
            rows.append({"tag": tag, "decision": "error", "source": d, "note": "no ckpt"})
            logger.warning(f"{tag}: no checkpoint under {d}"); continue
        try:
            tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                ck, torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()
            best = (-1, None, None)
            for v in VARIANTS:
                chunks = []
                with torch.no_grad():
                    for b in range(0, len(txt[v]), 32):
                        enc = tok(txt[v][b:b + 32], truncation=True, max_length=512,
                                  padding=True, return_tensors="pt").to("cuda")
                        chunks.append(model(**enc).logits.float().cpu().numpy())
                arr = np.concatenate(chunks).astype(np.float32)
                f1 = mf1(ytrue, arr.argmax(1))
                logger.info(f"  {tag} [{v}] F1={f1:.4f}")
                if f1 > best[0]:
                    best = (f1, v, arr)
            del model; torch.cuda.empty_cache()
            f1, v, arr = best
            if f1 < F1_FLOOR:
                rows.append({"tag": tag, "decision": "drop_lowf1", "self_f1": round(f1, 4),
                             "serialize": v, "source": ck})
                logger.warning(f"{tag}: best {f1:.4f} < {F1_FLOOR}"); continue
            np.savez(os.path.join(OUTDIR, f"{tag}.npz"), logits=arr)
            rows.append({"tag": tag, "decision": "ok", "self_f1": round(f1, 4), "clean": True,
                         "serialize": v, "source": ck, "note": "harvest_new variant-argmax"})
            logger.success(f"{tag}: {v} F1={f1:.4f} -> saved")
        except Exception as e:
            rows.append({"tag": tag, "decision": "error", "source": ck, "note": str(e)[:200]})
            logger.error(f"{tag}: {e}")

    with open(os.path.join(OUTDIR, "manifest.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.success(f"harvest_new: {sum(r['decision']=='ok' for r in rows)}/{len(NEW)} added")


if __name__ == "__main__":
    main()
