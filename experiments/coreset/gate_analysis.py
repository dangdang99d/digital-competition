"""gate_analysis.py — CAN the coreset-trained model rule out hard samples at inference?

The submission-relevant question behind drop-noisy (user 2026-07-09): beyond the +F1 headline,
does training on a denoised set make the model's OWN confidence a better gate — confident on
easy (non-hard) samples, unconfident on hard ones — so hard samples could be ruled out /
routed at inference?

One raw fp32 forward pass per model over the standard-split 14k val (NO calibration), tiers
fixed by the qwen3-champion ruler exactly as coreset_eval.py (§3 of results.md):
  easy    = qwen3-correct                       (label-trustworthy)
  suspect = qwen3 confident-wrong (MSP>0.6)     (suspect mislabel — F1 confounded)
  ambig   = qwen3 low-conf wrong                (genuine difficulty)

Per model (base + winners pvi06/aum06 + cl + drop-hard contrast cart15):
  - macro-F1 / acc per tier              (how it performs on easy vs hard)
  - MSP distribution per tier            (does confidence separate the tiers?)
  - AUROC  MSP -> ruler-hard  &  MSP -> own-error   (gate quality)
  - risk-coverage curve (abstain on lowest-MSP)     + oracle drop-all-ruler-hard point
  - suspect/ambig-tier prediction destination: ==label / ==qwen3-consensus / other

Logits cached to analysis/cache/coreset_gate_logits.npz (idempotent re-runs are free).
Figures -> experiments/coreset/figures/gate_*.png ; stats printed as markdown.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ocean/dacon \
    /home/ocean/miniconda3/envs/dacon/bin/python experiments/coreset/gate_analysis.py
"""
import argparse
import glob
import os

import numpy as np
from loguru import logger

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, split_indices

CACHE = "analysis/cache/coreset_gate_logits.npz"
FIGDIR = "experiments/coreset/figures"
RUN_GLOB = "output/pat/ft_*coreset_{}"
# palette.md categorical slots 1,2,3,5,6 (validated ordering; aqua/yellow get direct labels)
MODEL_COLOR = {"base": "#2a78d6", "pvi06": "#1baf7a", "aum06": "#eda100",
               "cl": "#4a3aa7", "cart15": "#e34948"}
TIER_COLOR = {"easy": "#2a78d6", "ambig": "#eda100", "suspect": "#e34948"}
BG = "#fcfcfb"


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def resolve_ckpt(run_dir):
    if os.path.exists(os.path.join(run_dir, "model.safetensors")):
        return run_dir
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    for c in reversed(cks):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    return None


def predict(tag, texts, bs, max_len):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    run_dir = (glob.glob(RUN_GLOB.format(f"rt_{tag}")) if tag != "base"
               else glob.glob(RUN_GLOB.format(tag)))
    assert len(run_dir) == 1, f"{tag}: expected 1 run dir, got {run_dir}"
    ck = resolve_ckpt(run_dir[0])
    tok = AutoTokenizer.from_pretrained(run_dir[0], trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ck, torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), bs):
            enc = tok(texts[b:b + bs], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            out.append(model(**enc).logits.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,pvi06,aum06,cl,cart15")
    ap.add_argument("--ruler", default="analysis/cache/qwen3_val_logits.npz")
    ap.add_argument("--suspect_msp", type=float, default=0.6)   # = coreset_eval default
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    tags = args.models.split(",")

    samples, y = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=42)
    texts = build_texts(samples)                                 # v1 = coreset retrain serialization
    yv = y_ids[va]

    # ---- tiers from the qwen3 ruler (identical to coreset_eval) ----
    qp = softmax(np.load(args.ruler)["logits"])
    q_pred, q_msp = qp.argmax(1), qp.max(1)
    q_acc = float((q_pred == yv).mean())
    assert 0.70 < q_acc < 0.82, f"ruler ordering mismatch (acc {q_acc:.4f})"
    easy = q_pred == yv
    suspect = ~easy & (q_msp > args.suspect_msp)
    ambig = ~easy & (q_msp <= args.suspect_msp)
    tiers = {"easy": easy, "ambig": ambig, "suspect": suspect}
    logger.info(f"tiers: easy {easy.sum()} · ambig {ambig.sum()} · suspect {suspect.sum()} "
                f"(ruler acc {q_acc:.4f})")

    # ---- logits per model (cached) ----
    cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    txv = [texts[i] for i in va]
    for t in tags:
        if t not in cache:
            logger.info(f"forward pass: {t}")
            cache[t] = predict(t, txv, args.batch_size, args.max_len)
            np.savez(CACHE, **cache)
    logger.info(f"logits cached -> {CACHE}")

    from sklearn.metrics import roc_auc_score
    stats, rc_curves = {}, {}
    coverages = np.arange(0.50, 1.0001, 0.02)
    for t in tags:
        p = softmax(cache[t])
        pred, msp = p.argmax(1), p.max(1)
        err = pred != yv
        s = {
            "overall": mf1(yv, pred), "acc": float((~err).mean()),
            "auroc_hard": roc_auc_score(~easy, -msp),            # MSP as a ruler-hard detector
            "auroc_err": roc_auc_score(err, -msp),               # MSP as an own-error detector
        }
        for name, m in tiers.items():
            s[f"f1_{name}"] = mf1(yv[m], pred[m])
            s[f"acc_{name}"] = float((pred[m] == yv[m]).mean())
            s[f"msp_{name}"] = float(msp[m].mean())
        # prediction destination on the hard tiers
        for name in ("suspect", "ambig"):
            m = tiers[name]
            s[f"{name}_to_label"] = float((pred[m] == yv[m]).mean())
            s[f"{name}_to_qwen3"] = float((pred[m] == q_pred[m]).mean())
        # risk-coverage on own MSP (keep the most-confident cov fraction)
        rc = []
        for cov in coverages:
            thr = np.quantile(msp, min(1.0, max(0.0, 1 - cov)))
            keep = msp >= thr
            rc.append(mf1(yv[keep], pred[keep]))
        rc_curves[t] = np.array(rc)
        # gate @ oracle coverage + the oracle itself (drop every ruler-hard sample)
        thr = np.quantile(msp, 1 - easy.mean())
        s["gate_at_easycov"] = mf1(yv[msp >= thr], pred[msp >= thr])
        s["oracle_easy"] = s["f1_easy"]
        stats[t] = s
        logger.success(f"{t}: overall {s['overall']:.4f} · easy {s['f1_easy']:.4f} · "
                       f"AUROC hard {s['auroc_hard']:.3f} / err {s['auroc_err']:.3f}")

    # ---- markdown tables ----
    print("\n## Gate analysis — per-tier performance (raw, qwen3-ruler tiers)\n")
    print("| model | overall F1 | easy F1 | ambig F1 | suspect F1 | easy acc | ambig acc | suspect acc |")
    print("|---|---|---|---|---|---|---|---|")
    for t in tags:
        s = stats[t]
        print(f"| {t} | {s['overall']:.4f} | {s['f1_easy']:.4f} | {s['f1_ambig']:.4f} | "
              f"{s['f1_suspect']:.4f} | {s['acc_easy']:.3f} | {s['acc_ambig']:.3f} | {s['acc_suspect']:.3f} |")
    print("\n## Confidence separation & gate quality (MSP, uncalibrated)\n")
    print("| model | MSP easy | MSP ambig | MSP suspect | AUROC hard | AUROC own-err | "
          "gate F1 @76.5% cov | oracle easy F1 |")
    print("|---|---|---|---|---|---|---|---|")
    for t in tags:
        s = stats[t]
        print(f"| {t} | {s['msp_easy']:.3f} | {s['msp_ambig']:.3f} | {s['msp_suspect']:.3f} | "
              f"{s['auroc_hard']:.3f} | {s['auroc_err']:.3f} | {s['gate_at_easycov']:.4f} | "
              f"{s['oracle_easy']:.4f} |")
    print("\n## Behavior on hard tiers — where do predictions go?\n")
    print("| model | suspect→label | suspect→qwen3-consensus | ambig→label | ambig→qwen3 |")
    print("|---|---|---|---|---|")
    for t in tags:
        s = stats[t]
        print(f"| {t} | {s['suspect_to_label']:.3f} | {s['suspect_to_qwen3']:.3f} | "
              f"{s['ambig_to_label']:.3f} | {s['ambig_to_qwen3']:.3f} |")

    _figures(tags, cache, yv, tiers, easy, stats, rc_curves, coverages)


def _figures(tags, cache, yv, tiers, easy, stats, rc_curves, coverages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- fig 1: MSP distribution per tier, one panel per model (small multiples) ---
    fig, axes = plt.subplots(1, len(tags), figsize=(3.1 * len(tags), 3.2),
                             sharey=True, sharex=True, facecolor=BG)
    bins = np.linspace(0, 1, 41)
    for ax, t in zip(np.atleast_1d(axes), tags):
        msp = softmax(cache[t]).max(1)
        ax.set_facecolor(BG)
        for name, m in tiers.items():
            ax.hist(msp[m], bins=bins, density=True, histtype="step", lw=2,
                    color=TIER_COLOR[name], label=name)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("max softmax (MSP)")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color="#e8e8e4", lw=0.7)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel("density")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Model confidence by val tier — does denoise training separate hard samples?",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/gate_msp_tiers.png", dpi=130, facecolor=BG, bbox_inches="tight")

    # --- fig 2: risk-coverage (macro-F1 of kept vs coverage) ---
    fig, ax = plt.subplots(figsize=(7.2, 5.0), facecolor=BG)
    ax.set_facecolor(BG)
    for t in tags:
        ax.plot(coverages, rc_curves[t], lw=2, color=MODEL_COLOR[t])
        ax.annotate(t, (coverages[-1], rc_curves[t][-1]), xytext=(6, 0),
                    textcoords="offset points", color=MODEL_COLOR[t],
                    fontsize=9, va="center")
    cov0 = easy.mean()
    ax.axvline(cov0, color="#9a9a92", lw=1, ls="--")
    ax.annotate(f"easy fraction ({cov0:.1%})", (cov0, ax.get_ylim()[0]),
                xytext=(4, 8), textcoords="offset points", fontsize=8.5, color="#6b6b64")
    for t in ("base", "pvi06"):
        if t in stats:
            ax.scatter([cov0], [stats[t]["oracle_easy"]], marker="*", s=140,
                       color=MODEL_COLOR[t], zorder=5)
    ax.annotate("★ = oracle (drop ALL ruler-hard)", (0.515, ax.get_ylim()[1]),
                fontsize=8.5, color="#6b6b64", va="top")
    ax.set_xlabel("coverage (fraction of val kept, most-confident first)")
    ax.set_ylabel("macro-F1 on kept samples (raw)")
    ax.set_title("Selective prediction — abstain on lowest-MSP samples")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color="#e8e8e4", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/gate_risk_coverage.png", dpi=130, facecolor=BG, bbox_inches="tight")

    # --- fig 3: suspect-tier prediction destination (stacked composition) ---
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(tags) + 1.6), facecolor=BG)
    ax.set_facecolor(BG)
    DEST = [("agrees w/ label", "#2a78d6"), ("agrees w/ qwen3 (≠label)", "#eda100"),
            ("other class", "#c9c9c2")]
    for k, t in enumerate(tags):
        s = stats[t]
        both = float((softmax(cache[t]).argmax(1)[tiers["suspect"]] == yv[tiers["suspect"]]).mean())
        v = [s["suspect_to_label"],
             s["suspect_to_qwen3"] - 0.0,   # qwen3-consensus (pred==qwen3, qwen3!=label on this tier)
             max(0.0, 1 - s["suspect_to_label"] - s["suspect_to_qwen3"])]
        left = 0.0
        for (lab, col), x in zip(DEST, v):
            ax.barh(k, x, left=left, color=col, height=0.6,
                    edgecolor=BG, linewidth=2)
            if x > 0.06:
                ax.text(left + x / 2, k, f"{x:.0%}", ha="center", va="center",
                        fontsize=9, color="#1a1a19")
            left += x
    ax.set_yticks(range(len(tags)), tags)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of suspect-mislabel val samples (n={})".format(int(tiers['suspect'].sum())))
    ax.set_title("On suspect-mislabel samples, what does each model predict?")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in DEST]
    ax.legend(handles, [l for l, _ in DEST], frameon=False, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/gate_suspect_behavior.png", dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"\nfigures -> {FIGDIR}/gate_msp_tiers.png · gate_risk_coverage.png · gate_suspect_behavior.png")


if __name__ == "__main__":
    main()
