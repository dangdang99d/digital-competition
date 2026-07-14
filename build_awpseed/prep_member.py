"""Prune a full-vocab granite checkpoint into an ensemble member dir (byte-identical
method to the shipped e38 pair/trio members — keep-list recovered from them and
verified byte-exact against member_0_e38_t031fd).

Usage (from repo root, build_awpseed/ present):
  python build_awpseed/prep_member.py --ckpt output/awp_seeds/ft_..._awpfd_s42 \
      --out build_awpseed/model/member_0_awpfd_s42 [--verify]

--verify runs 64 train rows through full AND pruned model and asserts argmax parity.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
EMB_KEY = "model.embeddings.tok_embeddings.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="full-vocab checkpoint dir")
    ap.add_argument("--out", required=True, help="member output dir")
    ap.add_argument("--variant", default="richargs")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    keep = torch.from_numpy(np.load(os.path.join(HERE, "keep_ids_ordered.npy")))
    sd = load_file(os.path.join(args.ckpt, "model.safetensors"))
    assert EMB_KEY in sd, f"{EMB_KEY} not in checkpoint"
    full_emb = sd[EMB_KEY]
    assert full_emb.shape[0] == 262152, f"unexpected vocab {full_emb.shape[0]} (already pruned?)"
    sd[EMB_KEY] = full_emb[keep].contiguous()

    os.makedirs(args.out, exist_ok=True)
    save_file(sd, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(HERE, "member_config_template.json")))
    # carry the label maps from the actual checkpoint (defensive; same for all our runs)
    src_cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    for k in ("id2label", "label2id"):
        if k in src_cfg:
            cfg[k] = src_cfg[k]
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    json.dump({"variant": args.variant},
              open(os.path.join(args.out, "serialize_variant.json"), "w"))
    print(f"member written: {args.out} (emb {tuple(sd[EMB_KEY].shape)}, fp16)")

    if args.verify:
        import sys
        sys.path.insert(0, os.path.dirname(HERE))
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.data import build_texts, load_samples
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        samples, _ = load_samples("./data")
        texts = build_texts(samples[:64], variant=args.variant)
        tok_full = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
        remap = torch.from_numpy(np.load(os.path.join(HERE, "model", "remap.npy"))).long().to(dev)
        mfull = AutoModelForSequenceClassification.from_pretrained(
            args.ckpt, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        mmem = AutoModelForSequenceClassification.from_pretrained(
            args.out, torch_dtype=torch.float16, local_files_only=True).to(dev).eval()
        enc = tok_full(texts, truncation=True, max_length=512, padding=True,
                       return_tensors="pt").to(dev)
        with torch.no_grad():
            lf = mfull(**enc).logits.float()
            enc_p = dict(enc)
            enc_p["input_ids"] = remap[enc["input_ids"]]
            lm = mmem(**enc_p).logits.float()
        agree = (lf.argmax(-1) == lm.argmax(-1)).float().mean().item()
        drift = (lf - lm).abs().max().item()
        print(f"VERIFY: argmax agreement {agree:.4f} (need 1.0), max|dlogit| {drift:.5f}")
        assert agree == 1.0, "PARITY FAIL — do not ship"
        print("PARITY OK")


if __name__ == "__main__":
    main()
