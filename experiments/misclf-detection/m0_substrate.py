"""M0 · Substrate extraction for the misclf-detection branch.

Runs the FROZEN base classifier (original qwen3 champion, 0.7682, plain CE) over ALL
70k train samples once, and caches per-sample:
  logits (N,14) · penultimate features h (N,1024) · label id · gen (sim/au) · step ·
  the seed-42 train/val split indices.
Everything M1/M2/M3/M5 consume this file — qwen3 is never in any detector's loop.

Penultimate features: Qwen3ForSequenceClassification pools the LAST non-pad token's
last-layer hidden state, then applies score (Linear 1024->14, no bias). We reconstruct
that pooled h from output_hidden_states and ASSERT  h @ score.Wt == the model's own
pooled logits (fp16 tol) on the first batches, so the tap point is verified, not assumed.

Run (repo root on PYTHONPATH):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/ocean/dacon \
    /home/ocean/miniconda3/envs/dacon/bin/python \
    experiments/misclf-detection/m0_substrate.py \
    --ckpt output/pat/ft_Qwen__Qwen3-Embedding-0.6B/checkpoint-10500 \
    --out analysis/cache/misclf_substrate_qwen3.npz
"""
import argparse
import re

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from src.data import CLASS_TO_ID, build_texts, load_samples, split_indices

_ID = re.compile(r"sess_([a-z]+)_.*-step_(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model weights+config (pruned submission dir)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B",
                    help="original HF qwen3 tokenizer")
    ap.add_argument("--remap", default=None, help="remap.npy: full-vocab id -> pruned embed row")
    ap.add_argument("--variant", default="v1", help="serialize variant the model trained on (e.g. richargs)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=512, help="length the model trained at")
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
    remap = None
    if args.remap:  # pruned embedding: map full-vocab tokenizer ids -> kept rows
        remap = torch.from_numpy(np.load(args.remap).astype(np.int64)).cuda()
        model.config.pad_token_id = int(remap[tok.pad_token_id].item())
    else:
        model.config.pad_token_id = tok.pad_token_id
    assert model.score.bias is None, "expected no-bias score head"
    W = model.score.weight.detach().float()  # (14, 1024)
    H = W.shape[1]

    samples, y = load_samples(args.data_dir)          # load order == train.jsonl order
    texts = build_texts(samples, variant=args.variant)  # match the model's training serialization
    N, C = len(texts), model.config.num_labels

    labels = np.array([CLASS_TO_ID[a] for a in y], dtype=np.int64)
    gen = np.array([_ID.match(s["id"]).group(1) for s in samples])          # 'sim'/'au'
    step = np.array([int(_ID.match(s["id"]).group(2)) for s in samples], dtype=np.int16)
    tr, va = split_indices(y, seed=42)                 # stratified 56k/14k, project standard

    logits = np.zeros((N, C), dtype=np.float32)
    feats = np.zeros((N, H), dtype=np.float16)

    order = np.argsort([len(t) for t in texts])        # length-sorted: fewer pad FLOPs
    max_recon_err = 0.0
    with torch.no_grad():
        for bi, i in enumerate(tqdm(range(0, N, args.batch_size), mininterval=5.0)):
            idx = order[i:i + args.batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to("cuda")
            if remap is not None:
                enc["input_ids"] = remap[enc["input_ids"]]
            out = model(**enc, output_hidden_states=True)
            lg = out.logits.float()                                    # (B,14) model-pooled
            hs = out.hidden_states[-1]                                 # (B,T,1024)
            # replicate the model's EXACT pooling: last token where id != pad_token_id
            ids = enc["input_ids"]
            non_pad = (ids != model.config.pad_token_id).int()
            last = (torch.arange(ids.shape[1], device=ids.device) * non_pad).argmax(1)
            h = hs[torch.arange(hs.shape[0], device=hs.device), last].float()  # (B,1024)
            # verify the tap: pooled h @ Wt must equal the model's own pooled logits
            recon = h @ W.t()
            err = (recon - lg).abs().max().item()
            max_recon_err = max(max_recon_err, err)
            if bi < 3:
                assert err < 0.5, f"pooling mismatch (max|recon-logits|={err:.3f}) — wrong tap point"
            logits[idx] = lg.cpu().numpy()
            feats[idx] = h.half().cpu().numpy()

    # parity vs the existing champion val cache (same checkpoint) — sanity, not required
    try:
        cached = np.load("analysis/cache/qwen3_val_logits.npz")["logits"]
        d = np.abs(logits[va] - cached).max()
        logger.info(f"parity vs qwen3_val_logits.npz: max|Δ|={d:.4f} on 14k val")
    except Exception as e:
        logger.warning(f"parity check skipped: {e}")

    val_acc = (logits[va].argmax(1) == labels[va]).mean()
    logger.info(f"val argmax acc {val_acc:.4f} · max pooling-recon err {max_recon_err:.4f}")
    np.savez(args.out, logits=logits, feats=feats, labels=labels,
             gen=gen, step=step, tr=tr.astype(np.int32), va=va.astype(np.int32))
    logger.success(f"substrate -> {args.out}  ({N} samples, feat dim {H})")


if __name__ == "__main__":
    main()
