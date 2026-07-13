"""Recover the remaining loadable standard-arch models skipped for `no_serialize`
(different-backbone diversity: base qwen3 + bge-m3 variants). Their serialize predates
the ft_results column, so try each plausible variant and KEEP THE BEST-SCORING one — a
wrong format scores far lower, so argmax recovers the true one without guessing. F1>=0.70 gate.
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
VARIANTS = ["v1", "richargs", "richmeta"]   # pre-column era + the two later formats

# tag -> dir   (all standard-split or full_data seed-42 -> hold out va_eval; clean)
CAND = {
    "qwen3_base":       "output/pat/ft_Qwen__Qwen3-Embedding-0.6B",
    "bgem3_hist0":      "output/pat/ft_BAAI__bge-m3_hist0",
    "bgem3_prune12":    "output/pat/ft_BAAI__bge-m3_prune12",
    "bgem3_rdrop":      "output/pat/ft_BAAI__bge-m3_rdrop",
    "bgem3_hardex":     "output/pat/ft_BAAI__bge-m3_hardex",
    "bgem3_v2_1024":    "output/ft_BAAI__bge-m3_v2_1024_full",
}


def resolve_ckpt(d):
    import glob
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
    assert np.array_equal(meta["va_idx"], va_eval), "slice drift!"
    ytrue = y_ids[va_eval]
    txt_cache = {v: build_texts([samples[i] for i in va_eval], variant=v) for v in VARIANTS}

    rows = []
    for tag, d in CAND.items():
        if os.path.exists(os.path.join(OUTDIR, f"{tag}.npz")):
            logger.info(f"{tag}: present, skip"); continue
        ck = resolve_ckpt(d)
        if ck is None:
            rows.append({"tag": tag, "decision": "error", "source": d, "note": "no ckpt"}); continue
        try:
            tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                ck, torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()
            best = (-1, None, None)
            for v in VARIANTS:
                txt = txt_cache[v]
                chunks = []
                with torch.no_grad():
                    for b in range(0, len(txt), 32):
                        enc = tok(txt[b:b + 32], truncation=True, max_length=512,
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
                             "serialize": v, "source": ck}); logger.warning(f"{tag}: best {f1:.4f} < floor")
                continue
            np.savez(os.path.join(OUTDIR, f"{tag}.npz"), logits=arr)
            rows.append({"tag": tag, "decision": "ok", "self_f1": round(f1, 4), "clean": True,
                         "serialize": v, "source": ck, "note": "recover2 variant-argmax"})
            logger.success(f"{tag}: {v} F1={f1:.4f} -> saved")
        except Exception as e:
            rows.append({"tag": tag, "decision": "error", "source": ck, "note": str(e)[:200]})
            logger.error(f"{tag}: {e}")

    with open(os.path.join(OUTDIR, "manifest.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.success(f"recover2: {sum(r['decision']=='ok' for r in rows)}/{len(CAND)} added")


if __name__ == "__main__":
    main()
