"""Top-1 vs top-2 UNCALIBRATED logit gap, correct vs incorrect, per model.

One figure per model (hist0 bge-m3, qwen3), two panels each: gap distribution on
correctly- and incorrectly-predicted val samples. The gap is the decision margin —
if wrong samples have systematically small gaps, margin-gated deferral works.

Caches full-val logits to analysis/cache/<model>_val_logits.npz.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

MODELS = {
    "hist0": dict(ckpt="output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500",
                  bias="output/pat/ft_BAAI__bge-m3_hist0/logit_bias.json",
                  title="hist0 (bge-m3 full-FT)", max_len=1024, pad_eos=False),
    "qwen3": dict(ckpt="output/pat/ft_Qwen__Qwen3-Embedding-0.6B/checkpoint-10500",
                  bias="output/pat/ft_Qwen__Qwen3-Embedding-0.6B/logit_bias.json",
                  title="Qwen3-0.6B full-FT", max_len=512, pad_eos=True),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLUE, RED = "#3987e5", "#d4553f"
INK, SUB, GRID, BG = "#0b0b0b", "#52514e", "#e4e2dd", "#fcfcfb"


@torch.no_grad()
def val_logits(cfg, cache):
    if os.path.exists(cache):
        return np.load(cache)["logits"]
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding)
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
    tok = AutoTokenizer.from_pretrained(cfg["ckpt"], local_files_only=True)
    if cfg["pad_eos"] and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["ckpt"], torch_dtype=torch.float16).to(DEV).eval()
    model.config.pad_token_id = tok.pad_token_id
    enc = [tok(texts[i], truncation=True, max_length=cfg["max_len"]) for i in va]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    coll = DataCollatorWithPadding(tokenizer=tok)
    out = np.zeros((len(enc), len(ALL_CLASSES)), dtype=np.float32)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        batch = {k: v.to(DEV) for k, v in coll([enc[i] for i in idx]).items()}
        lg = model(**batch).logits.float().cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = lg[j]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, logits=out)
    return out


def gap_fig(name, cfg, logits, y_true):
    # UNCALIBRATED by design: the logit bias is fit on this val set, so analyzing
    # errors through it would mix calibration overfit into the margin picture.
    lg = logits
    top2 = np.sort(lg, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    correct = lg.argmax(1) == y_true

    xmax = np.percentile(gap, 99.5)
    bins = np.linspace(0, xmax, 46)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), facecolor=BG, sharey=True)
    for ax, mask, color, label in ((axes[0], correct, BLUE, "correct"),
                                   (axes[1], ~correct, RED, "incorrect")):
        g = gap[mask]
        ax.set_facecolor(BG)
        ax.hist(g, bins=bins, color=color, zorder=3, density=True)
        med = np.median(g)
        ax.axvline(med, color=INK, lw=1.2, ls="--", zorder=4)
        ax.text(med + xmax * .015, ax.get_ylim()[1] * .0 + 0.02, f"median {med:.2f}",
                transform=ax.get_xaxis_transform(), fontsize=8.5, color=INK)
        ax.set_title(f"{label} (n={mask.sum():,}, {mask.mean():.1%} of val)",
                     fontsize=10.5, color=INK)
        ax.set_xlabel("top-1 − top-2 logit gap (uncalibrated)", fontsize=9, color=SUB)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(labelsize=8.5, colors=SUB)
    axes[0].set_ylabel("density", fontsize=9, color=SUB)
    fig.suptitle(f"{cfg['title']} — decision margin, correct vs incorrect (val n={len(gap):,})",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"figures/logit_gap_{name}.png"
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)

    # margin as a correctness signal
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(correct.astype(int), gap)
    print(f"{name}: median gap correct={np.median(gap[correct]):.2f} "
          f"wrong={np.median(gap[~correct]):.2f}  AUROC(gap->correct)={auroc:.3f}  saved -> {out}")


def main():
    samples, y = load_samples("./data")
    _, va = split_indices(y, seed=42)
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])
    for name, cfg in MODELS.items():
        logits = val_logits(cfg, f"analysis/cache/{name}_val_logits.npz")
        gap_fig(name, cfg, logits, y_true)


if __name__ == "__main__":
    main()
