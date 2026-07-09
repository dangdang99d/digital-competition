"""E24 method A — build reduced inputs from cached token scores at a given (scorer, keep-fraction).

Reads a_token_scores.npz; per example keeps the **top-k% highest-scoring CONTENT tokens**
(all special tokens always kept, original order preserved), and writes the reduced input_ids
in a pickle-free flat layout that finetune.py --reduced_ids consumes directly.

Output (rows aligned to full load_samples index 0..N-1):
  <out>/a_reduced_<scorer>_k<KK>.npz : ids_flat(int32, concat) lengths(int32, per-example)

No GPU, no shared code touched.  PYTHONPATH=. python experiments/token-selection/a_build_reduced.py --scorer attn --keep 0.9
"""
import argparse, os, numpy as np
from transformers import AutoTokenizer

CKPT = ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2"
        "_e8a_ls_richargs_full/checkpoint-8314")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="experiments/token-selection/artifacts/a_token_scores.npz")
    ap.add_argument("--scorer", choices=["attn", "sal"], required=True)
    ap.add_argument("--keep", type=float, required=True,
                    help="fraction of CONTENT tokens to keep, e.g. 0.9 / 0.8 / 0.7 / 0.6")
    ap.add_argument("--ckpt", default=CKPT, help="only for the special-token id set")
    ap.add_argument("--out", default="experiments/token-selection/artifacts")
    args = ap.parse_args()
    assert 0.0 < args.keep <= 1.0

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    special = set(int(x) for x in tok.all_special_ids)

    d = np.load(args.scores)
    ids_all, lens, score_all = d["ids"], d["lengths"], d[args.scorer]
    N = len(lens)
    out_len = np.zeros(N, dtype=np.int32)
    pieces = []
    for i in range(N):
        L = int(lens[i])
        ids = ids_all[i, :L]
        sc = score_all[i, :L].astype(np.float32)
        is_special = np.fromiter((int(t) in special for t in ids), dtype=bool, count=L)
        content = np.where(~is_special)[0]
        n_keep = int(np.ceil(args.keep * len(content))) if len(content) else 0
        keep_content = content[np.argsort(-sc[content])[:n_keep]]        # top-k% by score
        keep_pos = np.sort(np.concatenate([np.where(is_special)[0], keep_content]))
        red = ids[keep_pos].astype(np.int32)                            # order preserved
        pieces.append(red)
        out_len[i] = len(red)

    flat = np.concatenate(pieces).astype(np.int32) if pieces else np.zeros(0, np.int32)
    kk = int(round(args.keep * 100))
    path = os.path.join(args.out, f"a_reduced_{args.scorer}_k{kk}.npz")
    np.savez(path, ids_flat=flat, lengths=out_len)
    print(f"saved {path}  N={N}  mean kept frac (incl specials) = {out_len.sum()/lens.sum():.3f} "
          f"(target content-keep {args.keep})")


if __name__ == "__main__":
    main()
