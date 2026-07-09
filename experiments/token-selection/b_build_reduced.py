"""E24 method B — Phase 1 input builder: serialize all samples with the chosen fields DROPPED,
tokenize (champion tokenizer, max_len 512), write reduced input_ids for `finetune --reduced_ids`.

Same output format as method A (ids_flat + lengths, aligned to the 70k) → the SAME finetune hook.
A field-dropped serialization is just a different pre-tokenized input.

  # drop the 3 lowest-importance fields from Phase-0:
  PYTHONPATH=. python experiments/token-selection/b_build_reduced.py --drop meta.loc,meta.elapsed,meta.dirty --tag drop3
  # full-input anchor (drop nothing) -> should reproduce the champion when finetuned:
  PYTHONPATH=. python experiments/token-selection/b_build_reduced.py --tag full
"""
import argparse, os, numpy as np
from transformers import AutoTokenizer
from src.data import load_samples
from b_fields import apply_drops, ALL_OCCLUSIONS

CKPT = ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2"
        "_e8a_ls_richargs_full/checkpoint-8314")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--data", default="data")
    ap.add_argument("--drop", default="", help="comma-sep field names to DROP (empty = full richargs anchor)")
    ap.add_argument("--tag", required=True, help="label for the output file, e.g. drop3 / full")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="experiments/token-selection/artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    drops = [d.strip() for d in args.drop.split(",") if d.strip()]
    bad = [d for d in drops if d not in ALL_OCCLUSIONS]
    assert not bad, f"unknown fields {bad}; valid: {ALL_OCCLUSIONS}"

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    samples, labels = load_samples(args.data)
    N = len(samples) if not args.limit else min(args.limit, len(samples))
    texts = [apply_drops(samples[i], drops) for i in range(N)]
    enc = tok(texts, truncation=True, max_length=args.max_len, add_special_tokens=True)["input_ids"]
    lengths = np.array([len(e) for e in enc], dtype=np.int32)
    flat = np.concatenate([np.array(e, dtype=np.int32) for e in enc]) if N else np.zeros(0, np.int32)
    path = os.path.join(args.out, f"b_reduced_{args.tag}.npz")
    np.savez(path, ids_flat=flat, lengths=lengths)
    print(f"saved {path}  N={N}  dropped={drops or '(none/full)'}  mean len {lengths.mean():.0f} tok")


if __name__ == "__main__":
    main()
