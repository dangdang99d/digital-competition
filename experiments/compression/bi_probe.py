"""Block-Influence probe (ShortGPT): BI_i = 1 - mean_token cos(h_in, h_out) per layer.
Uses TRAIN rows (never the 3.5k held-out — layer selection must not see eval).
Prints ascending-BI prune order + keep-sets. Run on the model you will actually prune:
BI profiles are CHECKPOINT-SPECIFIC (t031's order applied to another AWP granite cost
-0.10 F1 at keep-18 vs that model's own choice — measured 2026-07-14).

  python -m experiments.compression.bi_probe --model_dir output/.../checkpoint-16628
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--serialize", default="richargs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--out", default="", help="optional .npy to save the BI vector")
    ap.add_argument("--remap", default="", help="remap.npy for vocab-pruned models")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.data import load_samples, build_texts, split_indices

    samples, y = load_samples(args.data_dir)
    texts = build_texts(samples, input_mode="context", max_hist=None, variant=args.serialize)
    tr, va = split_indices(np.asarray(y), seed=args.seed)  # strings — match finetune
    rng = np.random.default_rng(args.seed)
    pick = rng.permutation(tr)[:args.n]                     # TRAIN rows only
    texts = [texts[i] for i in pick]

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.truncation_side = "right"
    kw = dict(local_files_only=True, torch_dtype=torch.float16)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, attn_implementation="eager", reference_compile=False, **kw)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, attn_implementation="eager", **kw)
    model = model.to("cuda").eval()
    L = model.config.num_hidden_layers

    remap = None
    if args.remap:
        remap = torch.from_numpy(np.load(args.remap)).long()
        assert int(remap[tok.pad_token_id]) == model.config.pad_token_id

    bi_sum = np.zeros(L); ntok = 0
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            e = tok(texts[i:i + args.batch_size], truncation=True, max_length=args.max_len,
                    padding="max_length", return_tensors="pt")
            if remap is not None:
                e["input_ids"] = remap[e["input_ids"]]
            e = e.to("cuda")
            hs = model(**e, output_hidden_states=True).hidden_states
            m = e["attention_mask"].bool()
            for li in range(L):
                cos = F.cosine_similarity(hs[li][m].float(), hs[li + 1][m].float(), dim=-1)
                bi_sum[li] += (1.0 - cos).sum().item()
            ntok += int(m.sum().item())
    bi = bi_sum / ntok
    order = np.argsort(bi)

    print(f"\nBI per layer ({args.model_dir}, {len(texts)} train rows, {ntok} tokens):")
    for li in range(L):
        print(f"  L{li:2d}  BI={bi[li]:.4f}  {'#' * int(bi[li] / bi.max() * 40)}")
    print(f"\nascending-BI prune order: {order.tolist()}")
    for k in (2, 4, 6, 8):
        keep = sorted(set(range(L)) - set(order[:k].tolist()))
        print(f"  drop {k} -> keep {L-k}: {','.join(map(str, keep))}")
    if args.out:
        np.save(args.out, bi)


if __name__ == "__main__":
    main()
