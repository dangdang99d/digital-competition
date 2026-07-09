"""E24 method A — do the attention and saliency scorers rule out the SAME tokens?

User question (2026-07-09): if the tokens dropped by different scorers overlap, the
consensus-dropped set can be treated as model-independently unimportant and reused in
future runs (subject to an LB confirm against train-overfit).

CPU-only; reads the cached Phase-0 scores (a_token_scores.npz) — the drop sets are fully
determined by the scores + k, so this does NOT need the retrains to finish. For each
k ∈ {90,80,70,60}: per-sample bottom-(1-k) content tokens per scorer, then
  overlap = |drop_attn ∩ drop_sal| / |drop|   (chance level ≈ the drop fraction itself)
plus the top token STRINGS each scorer drops, and the consensus set size.

  PYTHONPATH=. python experiments/token-selection/a_drop_overlap.py
"""
import argparse
from collections import Counter

import numpy as np

SCORES = "experiments/token-selection/artifacts/a_token_scores.npz"
CKPT = ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2"
        "_e8a_ls_richargs_full/checkpoint-8314")


def drop_mask(score, ids, lengths, special, keep):
    """(N,L) bool: True where this scorer DROPS the token (bottom (1-keep) of content)."""
    N, L = score.shape
    s = score.astype(np.float32).copy()
    pos = np.arange(L)[None, :]
    valid = pos < lengths[:, None]
    content = valid & ~np.isin(ids, special)
    s[~content] = np.inf                          # specials/pad never droppable
    n_content = content.sum(1)
    n_drop = n_content - np.ceil(keep * n_content).astype(int)
    order = np.argsort(s, axis=1)                 # ascending: lowest score first
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, pos.repeat(N, 0) if False else np.broadcast_to(pos, (N, L)).copy(), axis=1)
    # rank[i, j] = position of token j in ascending score order
    return (rank < n_drop[:, None]) & content, n_content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.9, 0.8, 0.7, 0.6])
    ap.add_argument("--top", type=int, default=25, help="top dropped token strings to show")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    special = np.array(sorted(set(int(x) for x in tok.all_special_ids)), dtype=np.int32)

    d = np.load(args.scores)
    ids, lengths = d["ids"], d["lengths"]
    attn, sal = d["attn"], d["sal"]
    N = len(lengths)
    print(f"N={N}  total content tokens "
          f"{int((np.arange(ids.shape[1])[None, :] < lengths[:, None]).sum() - np.isin(ids, special).sum())}")

    print(f"\n| keep k | drop frac | overlap (∩/drop) | chance | lift | consensus tokens (∩) |")
    print("|---|---|---|---|---|---|")
    inter_masks = {}
    for keep in args.keeps:
        da, _ = drop_mask(attn, ids, lengths, special, keep)
        ds, ncont = drop_mask(sal, ids, lengths, special, keep)
        inter = da & ds
        n_drop = da.sum()
        ov = inter.sum() / max(n_drop, 1)
        chance = (da.sum(1) / np.maximum(ncont, 1)).mean()      # expected under independence
        print(f"| {keep:.0%} | {n_drop/ncont.sum():.3f} | {ov:.3f} | {chance:.3f} | "
              f"{ov/max(chance,1e-9):.1f}x | {int(inter.sum())} |")
        inter_masks[keep] = inter
        if keep == args.keeps[0]:
            da0, ds0 = da, ds

    # ---- what token strings get dropped (at the mildest k) ----
    k0 = args.keeps[0]
    def top_tokens(mask):
        c = Counter(ids[mask].tolist())
        return c
    ca, cs = top_tokens(da0), top_tokens(ds0)
    ci = top_tokens(inter_masks[k0])
    tot = Counter(ids[(np.arange(ids.shape[1])[None, :] < lengths[:, None]) & ~np.isin(ids, special)].tolist())

    def fmt(counter, n):
        rows = []
        for tid, cnt in counter.most_common(n):
            t = tok.convert_ids_to_tokens(int(tid)).replace("▁", "␣")
            rows.append(f"`{t}` {cnt/1000:.0f}k ({cnt/tot[tid]:.0%})")
        return rows

    print(f"\n### Top dropped token strings @ k={k0:.0%} (count, % of that token's occurrences dropped)")
    for name, c in (("attn", ca), ("sal", cs), ("consensus ∩", ci)):
        print(f"\n**{name}:** " + " · ".join(fmt(c, args.top)))

    # ---- per-sample agreement + rank correlation on a subsample ----
    rng = np.random.default_rng(0)
    sub = rng.choice(N, 2000, replace=False)
    from scipy.stats import spearmanr
    rhos = []
    for i in sub:
        L = int(lengths[i])
        m = ~np.isin(ids[i, :L], special)
        if m.sum() > 10:
            rhos.append(spearmanr(attn[i, :L][m], sal[i, :L][m]).statistic)
    print(f"\nper-sample Spearman(attn, sal) over content tokens: "
          f"median {np.median(rhos):.3f}  (IQR {np.percentile(rhos,25):.3f}–{np.percentile(rhos,75):.3f}, n=2000)")

    # ---- positional profile: where in the sequence do drops sit? ----
    relpos = np.arange(ids.shape[1])[None, :] / np.maximum(lengths[:, None], 1)
    for name, m in (("attn", da0), ("sal", ds0)):
        h, _ = np.histogram(relpos[m], bins=5, range=(0, 1))
        print(f"drop position profile ({name}, k={k0:.0%}, fifths of the sequence): "
              + " ".join(f"{x/m.sum():.0%}" for x in h))


if __name__ == "__main__":
    main()
