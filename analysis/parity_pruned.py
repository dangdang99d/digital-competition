"""Parity check: pruned-fp16 submission model vs the full fp32 checkpoint.

Replaces the missing src.verify_pruned. Serializes a val slice with the model's
trained variant, runs both models, and reports the argmax-prediction mismatch
rate. Pruning is only safe if this is ~0 (docstring claims prediction-identical).

Usage:
  python -m analysis.parity_pruned --full <ckpt_dir> --pruned <submission_model_dir> \
      --serialize richmeta [--n 3000]
"""
import argparse

import numpy as np
import torch

from src.data import (SERIALIZE_VARIANTS, build_texts, load_samples,
                      split_indices)


def predict(model_dir, texts, device, remap=None):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
    rm = (torch.from_numpy(np.load(f"{model_dir}/remap.npy")).long().to(device)
          if remap else None)
    enc = [tok(t, truncation=True, max_length=1024) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    coll = DataCollatorWithPadding(tokenizer=tok)
    out = [None] * len(enc)
    with torch.no_grad():
        for s in range(0, len(order), 64):
            idx = order[s:s + 64]
            batch = {k: v.to(device) for k, v in coll([enc[i] for i in idx]).items()}
            if rm is not None:
                batch["input_ids"] = rm[batch["input_ids"]]
            pr = model(**batch).logits.float().argmax(-1).cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = int(pr[j])
    del model
    torch.cuda.empty_cache()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--pruned", required=True)
    ap.add_argument("--serialize", default="v1", choices=sorted(SERIALIZE_VARIANTS))
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples, y = load_samples("./data")
    texts_all = build_texts(samples, input_mode="context", max_hist=None,
                            variant=args.serialize)
    _, va = split_indices(y, seed=42)
    va = va[:args.n]
    texts = [texts_all[i] for i in va]

    p_full = predict(args.full, texts, device, remap=False)
    p_prun = predict(args.pruned, texts, device, remap=True)
    mism = int((p_full != p_prun).sum())
    print(f"variant={args.serialize}  n={len(texts)}  mismatches={mism}  "
          f"rate={mism/len(texts):.4%}")
    if mism:
        bad = np.where(p_full != p_prun)[0][:10]
        print("  sample mismatches (full -> pruned):",
              [(int(p_full[i]), int(p_prun[i])) for i in bad])


if __name__ == "__main__":
    main()
