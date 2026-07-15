"""E39 arm B2 — LLM paraphrase (Qwen2.5-14B-Instruct via vLLM, bilingual + code-aware).

One call per sample: the model receives the sample's free-text spans as a numbered list and
returns a JSON array of paraphrases (same order, same count), preserving intent, language, and
all code identifiers/paths/numbers verbatim. Structure/labels untouched. Emits augmented JSONL.

Usage:
  python -m experiments.word_aug.gen_llm --targeting rare|blanket --out <path.jsonl>
      [--model Qwen/Qwen2.5-14B-Instruct-AWQ --tp 1 --max_model_len 4096]
GPU: 1×3090 (14B AWQ 4-bit ~10GB via vLLM). rare ~10-20 min, blanket ~1-2 h.
Needs: pip install vllm  (installed on the vast box, not in src/ requirements).
"""
import argparse
import json
import re

from loguru import logger

from experiments.word_aug.common import (apply_spans, augmentable_pool, get_freetext_spans,
                                         label_of, read_span, select_targets, write_augmented)

SYS = ("You rephrase short developer chat messages for data augmentation. Rules: keep the "
       "EXACT meaning and intent; keep the SAME language (Korean stays Korean, English stays "
       "English); copy every code identifier, file path, symbol name, number, and `backtick` "
       "span VERBATIM; vary only ordinary wording/phrasing. Return ONLY a JSON array of strings, "
       "same length and order as the input list, no commentary.")


def build_prompt(spans_text):
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(spans_text))
    return (f"Rephrase each of these {len(spans_text)} messages. Return a JSON array of "
            f"{len(spans_text)} strings in the same order.\n\n{numbered}")


def parse_array(text, n):
    """Extract a JSON array of n strings; return None on any mismatch."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(arr, list) and len(arr) == n and all(isinstance(x, str) for x in arr):
        return arr
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targeting", required=True, choices=["balanced", "blanket"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    ap.add_argument("--nshards", type=int, default=1, help="total parallel workers")
    ap.add_argument("--chunk", type=int, default=2000, help="rows per generate+write chunk")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="float16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90, seed=args.seed)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024, seed=args.seed)

    samples, y, pool = augmentable_pool()
    targets = select_targets(samples, y, pool, args.targeting, seed=args.seed)
    if args.limit:
        targets = targets[:args.limit]
    work = [(idx, c) for idx, k in targets for c in range(k)]
    work = work[args.shard::args.nshards]              # this worker's slice (data-parallel)
    logger.info(f"LLM {args.targeting} shard {args.shard}/{args.nshards}: {len(work)} copies")

    kept = skipped = 0
    with open(args.out, "w", encoding="utf-8") as fout:   # INCREMENTAL, chunked (crash-safe + monitorable)
        for start in range(0, len(work), args.chunk):
            batch = work[start:start + args.chunk]
            prompts, meta = [], []
            for idx, c in batch:
                s = samples[idx]
                texts = [read_span(s, sp_) for sp_ in get_freetext_spans(s)]
                msgs = [{"role": "system", "content": SYS},
                        {"role": "user", "content": build_prompt(texts)}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True))
                meta.append((idx, c, len(texts)))
            outs = llm.generate(prompts, sp)           # vLLM batches internally
            for (idx, c, n), o in zip(meta, outs):
                arr = parse_array(o.outputs[0].text, n)
                if arr is None:
                    skipped += 1
                    continue
                s = samples[idx]
                aug = apply_spans(s, arr)
                fout.write(json.dumps(
                    {"id": f"{s['id']}-llm{c}", "session_meta": aug["session_meta"],
                     "history": aug["history"], "current_prompt": aug["current_prompt"],
                     "label": label_of(samples, y, idx)}, ensure_ascii=False) + "\n")
                kept += 1
            fout.flush()
            logger.info(f"  {min(start + args.chunk, len(work))}/{len(work)} -> {kept} kept, {skipped} skipped")
    logger.success(f"LLM {args.targeting} shard {args.shard}: {kept} rows ({skipped} bad-JSON) -> {args.out}")


if __name__ == "__main__":
    main()
