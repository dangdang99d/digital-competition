"""E39 LLM via OpenAI — small correctness test before the batch run.
Runs a few fold-0 samples through gpt-4o-mini with the E39 paraphrase prompt, checks JSON
parses, structure preserved, code identifiers kept. Reads OPENAI_API_KEY from .env."""
import argparse
import os

import numpy as np

from experiments.word_aug.common import (apply_spans, augmentable_pool, get_freetext_spans,
                                         is_code_token, read_span)
from experiments.word_aug.gen_llm import SYS, build_prompt, parse_array


def load_key():
    for line in open(".env"):
        if line.startswith("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    load_key()
    from openai import OpenAI
    client = OpenAI()

    samples, y, pool = augmentable_pool()
    rng = np.random.default_rng(0)
    # sample diverse rows (varied labels, some with code-y prose)
    idxs = list(rng.choice(pool, size=args.n, replace=False))

    in_tok = out_tok = ok = 0
    for idx in idxs:
        s = samples[idx]
        spans = get_freetext_spans(s)
        texts = [read_span(s, sp_) for sp_ in spans]
        resp = client.chat.completions.create(
            model=args.model, temperature=0.7, max_tokens=512,
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": build_prompt(texts)}])
        out = resp.choices[0].message.content
        in_tok += resp.usage.prompt_tokens
        out_tok += resp.usage.completion_tokens
        arr = parse_array(out, len(texts))
        print(f"\n===== label={y[idx]}  spans={len(texts)}  JSON={'OK' if arr else 'FAIL'} =====")
        if arr is None:
            print("  RAW:", out[:300])
            continue
        ok += 1
        # code identifiers in originals that must survive
        codes = {w.strip("`'\".,!?);:") for t in texts for w in t.split() if is_code_token(w)}
        kept = [c for c in codes if all(c in a for a in [" ".join(arr)])]
        for o, a in zip(texts, arr):
            mark = "  " if o != a else " ="
            print(f" {mark}orig: {o[:105]}")
            print(f" {mark}aug : {a[:105]}")
        if codes:
            missing = [c for c in codes if c and c not in " ".join(arr)]
            print(f"   code tokens: {len(codes)} | missing after aug: {missing or 'none'}")

    print(f"\n=== {ok}/{args.n} parsed OK | tokens in={in_tok} out={out_tok} "
          f"| est full-job (138k rows): ${138236*(in_tok/args.n)*0.15/1e6 + 138236*(out_tok/args.n)*0.60/1e6:.1f} "
          f"(batch ~half) ===")


if __name__ == "__main__":
    main()
