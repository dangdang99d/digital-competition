"""E39 arm B1 — contextual MLM word substitution (XLM-RoBERTa-large, multilingual KO+EN).

For each targeted sample, mask a fraction of NON-code free-text words and let XLM-R predict
context-fitting replacements (top-k, sampled, excluding the original). Structure/code tokens
never masked. Emits augmented rows (JSONL) the E39 train loader unions with the originals.

Usage:
  python -m experiments.word-aug.gen_cmlm --targeting rare|blanket --out <path.jsonl>
      [--mask_p 0.15 --topk 5 --model FacebookAI/xlm-roberta-large --batch_size 64]
GPU: 1×3090 (560M model, fill-mask); rare ~minutes, blanket ~15-20 min.
"""
import argparse

import numpy as np
import torch
from loguru import logger

from experiments.word_aug.common import (apply_spans, augmentable_pool, get_freetext_spans,
                                         is_code_token, label_of, read_span, select_targets,
                                         write_augmented)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targeting", required=True, choices=["balanced", "blanket"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-large")
    ap.add_argument("--mask_p", type=float, default=0.10)   # EDA sweet spot (α=0.1); >0.1 hurts
    ap.add_argument("--min_cand", type=int, default=4,
                    help="skip masking spans with fewer than this many candidate words")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap targets (smoke)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    ap.add_argument("--nshards", type=int, default=1, help="total parallel workers")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, torch_dtype=torch.float16).cuda().eval()
    MASK = tok.mask_token

    samples, y, pool = augmentable_pool()
    targets = select_targets(samples, y, pool, args.targeting, seed=args.seed)
    if args.limit:
        targets = targets[:args.limit]
    targets = targets[args.shard::args.nshards]        # this worker's slice (data-parallel)
    logger.info(f"C-MLM {args.targeting} shard {args.shard}/{args.nshards}: {len(targets)} "
                f"target rows ({sum(k for _, k in targets)} copies) model={args.model}")

    rng = np.random.default_rng(args.seed + 1000 * args.shard)   # distinct noise per shard

    def _is_word(r):
        """A valid replacement token = has at least one alphanumeric/hangul char (not pure
        punctuation/special like '~', '.', '</s>')."""
        return any(c.isalnum() or "가" <= c <= "힣" for c in r)

    def substitute(text, rng):
        """Mask non-code words → XLM-R predicts replacements. Batched (one forward per text).
        mask_p over candidate words; skip spans with < min_cand candidates (short messages get
        proportionally over-perturbed); replacements filtered to real word tokens (no garbage)."""
        words = text.split(" ")
        cand = [j for j, w in enumerate(words) if w and not is_code_token(w) and len(w) > 1]
        if len(cand) < args.min_cand:                          # too short → leave unchanged
            return text
        k = max(1, int(round(len(cand) * args.mask_p)))
        pick = list(rng.choice(cand, size=min(k, len(cand)), replace=False))
        seqs = []
        for j in pick:
            m = list(words); m[j] = MASK; seqs.append(" ".join(m))
        enc = tok(seqs, return_tensors="pt", truncation=True, max_length=256,
                  padding=True).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits                       # [k, T, V]
        out = list(words)
        for bi, j in enumerate(pick):
            mpos = (enc.input_ids[bi] == tok.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mpos) == 0:
                continue
            top = torch.topk(logits[bi, mpos[0]], args.topk + 2).indices.tolist()
            reps = [tok.decode([t]).strip() for t in top]
            reps = [r for r in reps if r and r.lower() != words[j].lower()
                    and _is_word(r) and not is_code_token(r)]   # real words only (no '~', '.')
            if reps:
                out[j] = reps[int(rng.integers(0, len(reps)))]
        return " ".join(out)

    import json
    written = 0
    with open(args.out, "w", encoding="utf-8") as fout:      # INCREMENTAL write (crash-safe, monitorable)
        for n, (idx, ncopies) in enumerate(targets):
            s = samples[idx]
            spans = get_freetext_spans(s)
            for c in range(ncopies):
                new_texts = [substitute(read_span(s, sp), rng) for sp in spans]
                aug = apply_spans(s, new_texts)
                aug = {"id": f"{s['id']}-cmlm{c}", "session_meta": aug["session_meta"],
                       "history": aug["history"], "current_prompt": aug["current_prompt"],
                       "label": label_of(samples, y, idx)}
                fout.write(json.dumps(aug, ensure_ascii=False) + "\n")
                written += 1
            if (n + 1) % 200 == 0:
                fout.flush()
                logger.info(f"  {n + 1}/{len(targets)} target rows -> {written} augmented")
    logger.success(f"C-MLM {args.targeting}: wrote {written} augmented rows -> {args.out}")


if __name__ == "__main__":
    main()
