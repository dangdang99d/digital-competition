"""Which tokens are dead weight? Aggregate per-token saliency across the whole
val debug dump (output/val_debug_bgem3, --saliency all).

Per sample, each token's saliency is normalized by the sample's total, so every
sample contributes equally. Then:
  1. per token string: frequency, mean saliency share, length share — frequent
     tokens with near-zero mean share are removal candidates.
  2. per structural category (markers, separators, meta keys, numbers, ...):
     saliency share vs length share — a category taking 20% of tokens but 5% of
     saliency is boilerplate.

Usage: python -m analysis.token_importance [--dump output/val_debug_bgem3]
"""
import argparse
import glob
import json
import re
from collections import defaultdict

MARKERS = {"USER", "ACTION", "PROMPT", "ok", "s", "->"}
META_KEYS = {"tier", "lang", "turn", "budget", "ci", "dirty", "open", "pro", "free",
             "enterprise", "en", "ko", "mixed", "passed", "failed", "none", "True", "False"}
SEP_RE = re.compile(r"^[\s\[\]{}()'\":,=<>\-/.·;]+$")
NUM_RE = re.compile(r"^\s?\d+$")


def category(tok):
    t = tok.strip()
    if SEP_RE.match(tok) or t == "":
        return "separators/punct"
    if NUM_RE.match(tok):
        return "numbers"
    if t in MARKERS:
        return "structure markers (USER/ACTION/PROMPT/ok/->)"
    if t in META_KEYS:
        return "meta keys+values"
    return "content"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="output/val_debug_bgem3")
    ap.add_argument("--min_freq", type=int, default=2000)
    ap.add_argument("--top_k", type=int, default=25)
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    tok_stats = defaultdict(lambda: [0, 0.0])       # tok -> [count, sum norm saliency]
    cat_share = defaultdict(float)                  # category -> summed saliency share
    cat_len = defaultdict(int)                      # category -> token count
    n_samples, n_tokens = 0, 0

    for f in glob.glob(f"{args.dump}/*/*.json"):
        r = json.load(open(f))
        toks = r.get("tokens")
        if not toks:
            continue
        total = sum(v for _, v in toks) or 1.0
        n_samples += 1
        for t, v in toks:
            share = v / total
            s = tok_stats[t]
            s[0] += 1
            s[1] += share
            cat_share[category(t)] += share
            cat_len[category(t)] += 1
            n_tokens += 1

    print(f"aggregated {n_tokens/1e6:.1f}M tokens over {n_samples:,} samples\n")

    print("=== category level: saliency share vs length share ===")
    print(f"{'category':45s} {'len share':>10s} {'sal share':>10s} {'ratio':>6s}")
    for c in sorted(cat_share, key=lambda c: cat_share[c] / cat_len[c]):
        ls, ss = cat_len[c] / n_tokens, cat_share[c] / n_samples
        print(f"{c:45s} {ls:10.1%} {ss:10.1%} {ss/ls:6.2f}")

    print(f"\n=== most frequent tokens with LOWEST mean saliency share (freq >= {args.min_freq}) ===")
    rows = [(s[1] / s[0], t, s[0]) for t, s in tok_stats.items() if s[0] >= args.min_freq]
    rows.sort()
    print(f"{'mean share/token':>16s}  {'freq':>7s}  token")
    for ms, t, n in rows[: args.top_k]:
        print(f"{ms:16.6f}  {n:7d}  {t!r}")

    print(f"\n=== highest mean saliency share (for contrast) ===")
    for ms, t, n in rows[-8:]:
        print(f"{ms:16.6f}  {n:7d}  {t!r}")


if __name__ == "__main__":
    main()
