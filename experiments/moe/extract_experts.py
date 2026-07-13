"""E33 follow-up: disassemble a trained GraniteMoE into standalone granite
classifiers — one per expert — so they're reusable as ordinary members in any
ensemble / OOF experiment (loadable with from_pretrained, same as a normal
granite full-FT checkpoint).

Each extracted model = shared embedding + shared trunk (layers 0..k-1) + that
expert's layers (k..21) + expert final_norm/head/classifier = a complete 22-layer
ModernBertForSequenceClassification. By construction it reproduces the expert's
solo output (verified live: the MoE's per-expert path IS this standalone forward,
real-granite parity max|Δlogp|=0 in the E33 smoke). A --parity_rows check
re-confirms on real holdout rows.

By default extracts only "alive" experts (solo macro-F1 on the cached holdout
>= --min_solo), since E33 leaves 1-2 dead experts per MoE (solo down to 0.26).

Usage:
  python -m experiments.moe.extract_experts --run_dir output/moe_..._e33_moe_k6
  # or a specific set:  --experts 2,3   |  keep all:  --min_solo 0
"""
import argparse
import json
import os

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples
from src.moe_model import load_moe


def mf1(probs, y):
    return f1_score(y, probs.argmax(-1), labels=list(range(len(ALL_CLASSES))),
                    average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="a trained MoE run dir (moe_state.pt + moe_config.json + config.json)")
    ap.add_argument("--out_root", default="output/moe_experts")
    ap.add_argument("--min_solo", type=float, default=0.65,
                    help="extract experts with cached holdout solo macro-F1 >= this (0 = all)")
    ap.add_argument("--experts", default="", help="explicit comma list of expert idxs (overrides --min_solo)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--serialize", default="richargs")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--parity_rows", type=int, default=16, help="holdout rows to re-verify extraction (0=skip)")
    ap.add_argument("--save_dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_tag = os.path.basename(args.run_dir.rstrip("/")).split("e33_moe_")[-1]
    meta = json.load(open(os.path.join(args.run_dir, "moe_config.json")))
    k, n_exp = meta["trunk_k"], meta["n_experts"]
    logger.info(f"MoE {run_tag}: trunk_k={k} n_experts={n_exp}")

    moe = load_moe(args.run_dir, device="cpu", torch_dtype=torch.float32).eval()

    # which experts?
    parts = os.path.join(args.run_dir, "moe_val_parts.npz")
    solo = None
    if os.path.exists(parts):
        d = np.load(parts)
        P = np.exp(d["expert_logprobs"]); y = d["labels"]
        solo = np.array([mf1(P[:, e], y) for e in range(n_exp)])
        logger.info(f"cached solo macro-F1: {[round(float(s),4) for s in solo]}")
    if args.experts:
        idxs = [int(x) for x in args.experts.split(",") if x.strip() != ""]
    elif solo is not None:
        idxs = [e for e in range(n_exp) if solo[e] >= args.min_solo]
        logger.info(f"alive experts (solo>={args.min_solo}): {idxs} "
                    f"(dropped {[e for e in range(n_exp) if e not in idxs]})")
    else:
        idxs = list(range(n_exp))

    # base 22-layer granite config (saved with the run) + tokenizer
    cfg = AutoConfig.from_pretrained(args.run_dir, num_labels=len(ALL_CLASSES),
                                     id2label={i: c for i, c in enumerate(ALL_CLASSES)},
                                     label2id={c: i for i, c in enumerate(ALL_CLASSES)})
    cfg._attn_implementation = "sdpa"
    full_depth = cfg.num_hidden_layers
    assert full_depth == k + len(moe.experts[0].layers), \
        f"depth mismatch: cfg {full_depth} vs trunk {k} + expert {len(moe.experts[0].layers)}"
    tok = AutoTokenizer.from_pretrained(args.run_dir, trust_remote_code=True)

    # tokenize the FULL 3,500 holdout once (reuse cached absolute indices + labels);
    # used for parity AND for dumping each expert's raw logits for later ensemble access
    va = labels_va = enc_ids = enc_mask = exp_lp_ref = None
    if os.path.exists(parts):
        dd = np.load(parts)
        va, labels_va = dd["va_idx"], dd["labels"]
        samples, _ = load_samples(args.data_dir)
        texts = build_texts(samples, variant=args.serialize)
        toks = tok([texts[i] for i in va], truncation=True, max_length=args.max_len,
                   padding=False)
        enc_ids, enc_mask = toks["input_ids"], toks["attention_mask"]
        if args.parity_rows:  # reference logprobs from the MoE itself, first rows
            r = min(args.parity_rows, len(va))
            pe = tok([texts[i] for i in va[:r]], truncation=True, max_length=args.max_len,
                     padding=True, return_tensors="pt")
            with torch.no_grad():
                _, exp_lp_ref = moe.predict_parts(pe["input_ids"], pe["attention_mask"])

    @torch.no_grad()
    def dump_logits(model):
        """raw pre-softmax logits over the full holdout, dynamic-padded batches."""
        model.to(device)
        out = np.zeros((len(enc_ids), len(ALL_CLASSES)), dtype=np.float32)
        bs = 32
        for s in range(0, len(enc_ids), bs):
            batch = tok.pad({"input_ids": enc_ids[s:s + bs],
                             "attention_mask": enc_mask[s:s + bs]}, return_tensors="pt")
            lg = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits.float()
            out[s:s + len(lg)] = lg.cpu().numpy()
        model.to("cpu")
        return out

    save_dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.save_dtype]
    os.makedirs(args.out_root, exist_ok=True)
    manifest = []
    all_logits, all_names = [], []
    for e in idxs:
        m = AutoModelForSequenceClassification.from_config(cfg)
        # shared embedding + trunk (0..k-1)
        m.model.embeddings.load_state_dict(moe.trunk.embeddings.state_dict())
        for i in range(k):
            m.model.layers[i].load_state_dict(moe.trunk.layers[i].state_dict())
        # expert layers (k..depth-1) + expert final_norm/head/classifier
        for j, i in enumerate(range(k, full_depth)):
            m.model.layers[i].load_state_dict(moe.experts[e].layers[j].state_dict())
        m.model.final_norm.load_state_dict(moe.experts[e].final_norm.state_dict())
        m.head.load_state_dict(moe.experts[e].head.state_dict())
        m.classifier.load_state_dict(moe.experts[e].classifier.state_dict())
        m.config.pad_token_id = tok.pad_token_id
        m.eval()

        out = os.path.join(args.out_root, f"{run_tag}_e{e}")
        s = float(solo[e]) if solo is not None else float("nan")

        # ---- logits on the full 3,500 holdout (fp32, before the fp16 save-cast) ----
        if enc_ids is not None:
            logits = dump_logits(m)                                  # (3500, 14)
            os.makedirs(out, exist_ok=True)
            np.savez(os.path.join(out, "val_logits_3500.npz"),
                     logits=logits, va_idx=va, labels=labels_va,
                     src_moe=run_tag, expert=e, solo_mf1=s)
            check = mf1(logits, labels_va)                           # self-consistency
            if exp_lp_ref is not None:                               # parity vs MoE path
                lp = np.log(np.exp(logits[:len(exp_lp_ref)])
                            / np.exp(logits[:len(exp_lp_ref)]).sum(-1, keepdims=True))
                dmax = float(np.abs(lp - exp_lp_ref[:, e].numpy()).max())
                assert dmax < 1e-3, f"expert {e} parity FAIL max|Δlogp|={dmax}"
                logger.info(f"expert {e}: parity OK (max|Δlogp|={dmax:.1e}), "
                            f"logit-F1 {check:.4f} vs cached {s:.4f}")
            all_logits.append(logits); all_names.append(f"{run_tag}_e{e}")

        # ---- save the standalone checkpoint ----
        if save_dt != torch.float32:
            m.to(save_dt)
        m.config.torch_dtype = save_dt
        m.save_pretrained(out)
        tok.save_pretrained(out)
        manifest.append({"src_moe": run_tag, "expert": e, "solo_mf1": round(s, 4), "dir": out})
        logger.success(f"extracted {run_tag}_e{e} (solo {s:.4f}) -> {out} ({args.save_dtype})")

    # combined logits file for easy ensemble access (append-safe per MoE)
    if all_logits:
        comb = os.path.join(args.out_root, f"val_logits_3500_{run_tag}.npz")
        np.savez(comb, logits=np.stack(all_logits), names=np.array(all_names),
                 va_idx=va, labels=labels_va)
        logger.info(f"combined logits ({len(all_logits)} experts) -> {comb}")

    mf = os.path.join(args.out_root, "manifest.jsonl")
    with open(mf, "a") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    logger.info(f"appended {len(manifest)} experts -> {mf}")


if __name__ == "__main__":
    main()
