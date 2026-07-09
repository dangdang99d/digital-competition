"""E24 method A — Phase 0: score every token of every sample with the FROZEN champion.

Two per-token importance signals, computed once and cached:
  - attn : attention RECEIVED  (mean over layers, heads, non-pad query positions)   [A1]
  - sal  : gradient saliency |dL/demb . emb| under the TRUE-label CE                 [A2]

Serialization / tokenizer / max_len match the champion's training EXACTLY (richargs,
max_len 512, truncation) so token positions align 1:1 with what Phase-1 training sees.

Output (rows aligned to the full load_samples index 0..N-1):
  <out>/a_token_scores.npz : ids(int32 N,L) lengths(int32 N) attn(f16 N,L) sal(f16 N,L)

Standalone: loads the champion read-only, touches NO shared code, ships nothing.
  python -m experiments... no — run:  PYTHONPATH=. python experiments/token-selection/a_score_tokens.py
"""
import argparse, os, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data import load_samples, build_texts, CLASS_TO_ID, ALL_CLASSES

CKPT = ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2"
        "_e8a_ls_richargs_full/checkpoint-8314")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--data", default="data")
    ap.add_argument("--variant", default="richargs")       # match champion serialization
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: score only first N (load order)")
    ap.add_argument("--out", default="experiments/token-selection/artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ckpt, num_labels=len(ALL_CLASSES), attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True).to(dev).eval()
    assert getattr(model.config, "classifier_pooling", "cls") == "cls", "expected CLS pooling"
    # NOTE: do NOT freeze params — freezing all of them zeros the input-embedding gradient
    # on this model (saliency -> 0). We just avoid stepping any optimizer; zero_grad per batch.

    samples, labels = load_samples(args.data)
    N = len(samples) if not args.limit else min(args.limit, len(samples))
    texts = build_texts(samples[:N], variant=args.variant)
    y = np.array([CLASS_TO_ID[labels[i]] for i in range(N)], dtype=np.int64)

    # pre-tokenize (no pad) -> per-sample ids + lengths; length-sort to minimize padding
    enc = tok(texts, truncation=True, max_length=args.max_len,
              add_special_tokens=True)["input_ids"]
    lens = np.array([len(e) for e in enc], dtype=np.int32)
    order = np.argsort(lens)
    L = int(lens.max())
    print(f"N={N}  Lmax={L}  median_len={int(np.median(lens))}")

    ids_out = np.zeros((N, L), dtype=np.int32)
    attn_out = np.zeros((N, L), dtype=np.float16)
    sal_out = np.zeros((N, L), dtype=np.float16)
    emb_layer = model.get_input_embeddings()
    pad_id = tok.pad_token_id

    from tqdm.auto import tqdm
    for b in tqdm(range(0, N, args.batch_size), desc="scoring", mininterval=5.0):
        bi = order[b:b + args.batch_size]
        chunk = [enc[i] for i in bi]
        m = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), m), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), m), dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        ids, att = ids.to(dev), att.to(dev)
        mask = att.float()

        # --- A1: attention received (no grad) ---
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att, output_attentions=True)
            recv = torch.zeros_like(mask)
            for A in out.attentions:                       # (B,H,S,S): query i -> key j
                aq = (A.mean(1) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                recv += aq                                 # received = mean over non-pad queries
            recv = (recv / len(out.attentions)) * mask
        del out

        # --- A2: gradient saliency (true-label CE) ---
        model.zero_grad(set_to_none=True)
        embeds = emb_layer(ids).detach().clone().requires_grad_(True)
        logits = model(inputs_embeds=embeds, attention_mask=att).logits
        loss = F.cross_entropy(logits, torch.tensor(y[bi], device=dev))
        loss.backward()
        sal = ((embeds.grad * embeds).sum(-1).abs() * mask)

        recv = recv.cpu().numpy()
        sal = sal.detach().cpu().float().numpy()
        for j, i in enumerate(bi):
            Lj = len(chunk[j])
            a, s = recv[j, :Lj], sal[j, :Lj]
            ids_out[i, :Lj] = np.array(chunk[j], dtype=np.int32)
            # normalize per-example to [0,1] (ranking-preserving) so float16 storage is safe
            attn_out[i, :Lj] = (a / (a.max() + 1e-12)).astype(np.float16)
            sal_out[i, :Lj] = (s / (s.max() + 1e-12)).astype(np.float16)

    path = os.path.join(args.out, "a_token_scores.npz")
    np.savez(path, ids=ids_out, lengths=lens, attn=attn_out, sal=sal_out)
    print(f"saved {path}  size~{os.path.getsize(path)/1e6:.0f}MB")


if __name__ == "__main__":
    main()
