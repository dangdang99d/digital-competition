"""E40 diagnostic — apply the char-noise augmentation to the VALIDATION data and measure the
best model's macro-F1 drop. Inference only (no training). Shows how sensitive our model is to
surface noise = the robustness angle the lit flagged (char-noise aug helps THIS, not clean test).

Models:
  AWP  (output/e34/vast_r1/e34_c_awp)  — best SINGLE model, LB 0.78557; richargs; val CONTAMINATED
        (full_data) so absolute F1 inflated, but the clean-vs-noised DROP is the read.
  e9   (e9_granite_ls, v1)             — HONEST clean 14k val; weaker but the drop is uncontaminated.
All val rows noised (alpha=1.0) at each p, fixed seed for reproducibility.
"""
import argparse
import json
import re

import numpy as np
import torch
from loguru import logger

from src.data import (CLASS_TO_ID, SERIALIZE_VARIANTS, load_samples, macro_f1,
                      serialize, split_indices)
from experiments.augmentation.text_noise import noise_sample_freetext

MODELS = {
    "AWP_best_single": ("output/e34/vast_r1/e34_c_awp", "richargs", "val* CONTAMINATED"),
    "e9_honest":       ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-"
                        "r2_e9_granite_ls/checkpoint-10500", "v1", "HONEST clean val"),
}
PS = [0.0, 0.02, 0.05, 0.10]


def is_ko(s):
    return sum("가" <= c <= "힣" for c in (s or "")) > 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    samples, y = load_samples("./data")
    y_ids_all = np.array([CLASS_TO_ID[t] for t in y])
    _, va = split_indices(y, seed=42)
    # HONEST 3.5k slice = the full_data held-out eval slice (finetune.py:1241):
    # 25% of the 14k val, stratified, seed 42 — held out of every --full_data run (AWP incl.)
    from sklearn.model_selection import train_test_split
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids_all[va], random_state=42)
    va = va_eval
    logger.info(f"eval slice = full_data 3.5k held-out (n={len(va)})")
    val = [samples[i] for i in va]
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])
    ko_mask = np.array([is_ko(s.get("current_prompt")) for s in val])

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def fwd(model, tok, texts):
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), model.config.num_labels), np.float32)
        with torch.no_grad():
            for i in range(0, len(order), args.batch_size):
                idx = order[i:i + args.batch_size]
                enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                          padding=True, return_tensors="pt").to("cuda")
                out[idx] = model(**enc).logits.float().cpu().numpy()
        return out.argmax(1)

    results = {}
    for tag, (ckpt, variant, note) in MODELS.items():
        logger.info(f"\n===== {tag}  ({variant}, {note}) =====")
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
        model.config.pad_token_id = tok.pad_token_id
        kw = SERIALIZE_VARIANTS[variant]
        base = None
        for p in PS:
            rng = np.random.default_rng(1234)   # fixed → reproducible noise per p
            vs = val if p == 0.0 else [noise_sample_freetext(s, rng, p) for s in val]
            texts = [serialize(s, **kw) for s in vs]
            pred = fwd(model, tok, texts)
            f1 = macro_f1(y_true, pred)
            f1_ko = macro_f1(y_true[ko_mask], pred[ko_mask])
            f1_en = macro_f1(y_true[~ko_mask], pred[~ko_mask])
            if p == 0.0:
                base = f1
            d = f1 - base
            logger.info(f"  p={p:<4}  macroF1={f1:.4f}  (Δ{d:+.4f})  | KO {f1_ko:.4f}  EN {f1_en:.4f}")
            results[f"{tag}_p{p}"] = dict(f1=f1, d=d, ko=f1_ko, en=f1_en)
        del model
        torch.cuda.empty_cache()

    with open("experiments/augmentation/e40_val_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.success("wrote experiments/augmentation/e40_val_robustness.json")


if __name__ == "__main__":
    main()
