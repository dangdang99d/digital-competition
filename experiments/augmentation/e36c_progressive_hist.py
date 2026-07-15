"""E36 arm A (user request) — PROGRESSIVE-HISTORY multi-view averaging.

For each val row, build views hist0, hist1, ..., histH (H = that row's full history
length): hist_j = the input with history truncated to its last j events (hist0 = zero
history, kept meta+prompt). Softmax-average the model's outputs across j=0..H, argmax,
macro-F1. Compare vs the standard single full-history inference.

Two models:
  - e9_granite_ls (v1+LS, split-trained)  -> HONEST 14k val (headline)
  - e8a_ls_richargs_full                  -> champion; val is full_data-CONTAMINATED
    (absolute F1 inflated; relative avg-vs-single signal only)
Histories in this data cap at 12 events, so j in 0..12 covers everything.
"""
import argparse
import numpy as np
import torch
from loguru import logger
from src.data import (CLASS_TO_ID, build_texts, load_samples, macro_f1,
                      split_indices)

MODELS = {
    "e9_v1_honest": ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-"
                     "r2_e9_granite_ls/checkpoint-10500", "v1"),
    "e8a_richargs_champ": ("output/pat/ft_ibm-granite__granite-embedding-311m-"
                           "multilingual-r2_e8a_ls_richargs_full/checkpoint-8314",
                           "richargs"),
}
MAXJ = 12  # history caps at 12 events in this dataset


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def run_model(ckpt, variant, val_samples, y_true, max_len, bs):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
    model.config.pad_token_id = tok.pad_token_id

    def fwd(texts):
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), model.config.num_labels), np.float32)
        with torch.no_grad():
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt").to("cuda")
                out[idx] = model(**enc).logits.float().cpu().numpy()
        return out

    H = np.array([min(len(s["history"]), MAXJ) for s in val_samples])  # per-row full depth
    # softmax at every truncation depth j
    P = np.zeros((MAXJ + 1, len(val_samples), model.config.num_labels), np.float32)
    for j in range(MAXJ + 1):
        if j == 0:
            texts = build_texts(val_samples, variant=variant, strip_history=True)
        else:
            texts = build_texts(val_samples, variant=variant, max_hist=j)
        P[j] = softmax(fwd(texts))
        logger.info(f"  depth hist{j:<2d} single-view F1 = "
                    f"{macro_f1(y_true, P[j].argmax(1)):.4f}")

    # per-row averaging over a depth range [floor(H), H]; floor set by scheme
    def avg_range(floor_fn):
        acc = np.zeros_like(P[0]); cnt = np.zeros(len(val_samples))
        for j in range(MAXJ + 1):
            lo = floor_fn(H)                      # per-row lower bound (array)
            m = (j >= lo) & (j <= H)              # include depth j for these rows
            acc[m] += P[j][m]; cnt[m] += 1
        # first-step rows (H=0) always covered by j=0
        return macro_f1(y_true, (acc / cnt[:, None]).argmax(1))

    full_f1 = macro_f1(y_true, P[H, np.arange(len(H))].argmax(1))  # standard inference
    schemes = {
        "single full-history (baseline)": full_f1,
        "avg hist0..histH (literal)":     avg_range(lambda H: np.zeros_like(H)),
        "avg hist1..histH (drop hist0)":  avg_range(lambda H: np.minimum(1, H)),
        "avg histceil(H/2)..histH":       avg_range(lambda H: (H + 1) // 2),
        "avg last-3 depths {H-2..H}":     avg_range(lambda H: np.maximum(0, H - 2)),
    }
    return full_f1, schemes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    val_samples = [samples[i] for i in va]
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])

    for tag, (ckpt, variant) in MODELS.items():
        note = "HONEST val" if "honest" in tag else "val CONTAMINATED (relative only)"
        logger.info(f"\n===== {tag}  ({variant})  [{note}] =====")
        base, schemes = run_model(ckpt, variant, val_samples, y_true,
                                  args.max_len, args.batch_size)
        logger.info(f"  --- {tag} results ---")
        for name, f1 in schemes.items():
            d = f1 - base
            flag = "  <== beats" if d > 1e-9 else ""
            logger.info(f"  {name:34s} F1 = {f1:.4f}  (Δ {d:+.4f}){flag}")


if __name__ == "__main__":
    main()
