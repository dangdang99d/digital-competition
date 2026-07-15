"""E39 arm B3 — LLM paraphrase via a LOCAL vLLM model (default: Qwen3-30B-A3B-Instruct-2507-FP8).

Offline in-process vLLM (no server): loads the model once, batches internally, writes augmented
JSONL incrementally. Data-parallel via --shard/--nshards (one process per GPU-group). Reuses the
exact prompt + span machinery as the OpenAI arm, so B3 (local Qwen) is directly comparable to /
combinable with B2 (gpt-4o-mini). Distinct id suffix (default `qw`) keeps the two arms separate.

WHY THIS EXISTS / how it dodges the prior failure (Qwen2.5-14B-AWQ -> `!!!!` garbage):
  - the checkpoint is **FP8** (native Blackwell tensor cores), NOT the AWQ int4 kernel that broke;
  - **dtype='auto'** lets vLLM read the FP8 quant from config (the old code hardcoded float16);
  - **--smoke** runs 8 fixed probes through the SAME load+generate path and ASSERTS the output is
    sane (coherent, code identifiers verbatim, valid JSON, no repeated-char garbage) BEFORE any
    real generation. Fan-out only proceeds if smoke exits 0. bf16 fallback: pass --dtype bfloat16.

  # gate (one GPU-group), exits non-zero on garbage:
  python -m experiments.word_aug.gen_llm_vllm --smoke --model <path> --tp 2
  # real shard:
  python -m experiments.word_aug.gen_llm_vllm --targeting balanced --out <p> --model <path> \
      --tp 2 --shard 0 --nshards 4
"""
import argparse
import json
import re
import sys

from loguru import logger

from experiments.word_aug.common import (apply_spans, augmentable_pool, get_freetext_spans,
                                         label_of, read_span, select_targets)
from experiments.word_aug.gen_llm import SYS, build_prompt, parse_array

DEFAULT_MODEL = "output/models/Qwen3-30B-A3B-Instruct-2507-FP8"

# (span-text list, [tokens that MUST survive verbatim]) — probes span KO / EN / code / mixed.
SMOKE_PROBES = [
    (["버튼 컴포넌트에 로딩 상태를 추가해줘"], []),
    (["Add a loading spinner to the submit button"], []),
    (["Refactor `getUserData()` in src/api/user.ts to use the new fetchClient"],
     ["getUserData", "src/api/user.ts", "fetchClient"]),
    (["`config.yaml`의 timeout 값을 30초로 늘려줘"], ["config.yaml", "timeout", "30"]),
    (["파일을 저장할 수 없어요", "Try running npm install first"], ["npm install"]),
    (["Change the port from 8080 to 3000 in docker-compose.yml"],
     ["8080", "3000", "docker-compose.yml"]),
    (["데이터베이스 마이그레이션 스크립트가 실패하는데 원인을 찾아줘"], []),
    (["Use `useMemo` to memoize the `filteredItems` array"], ["useMemo", "filteredItems"]),
]


def _is_garbage(s):
    """Detect the prior failure signature: a string dominated by one repeated char (e.g. '!!!!')."""
    t = "".join(s.split())
    if len(t) < 4:
        return False
    top = max(t.count(c) for c in set(t))
    return top / len(t) > 0.6


def _render(tok, texts):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": build_prompt(texts)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:                                   # template without the kwarg (non-thinking)
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_smoke(llm, tok, sp):
    """Return True iff all probes yield sane, identifier-preserving, non-garbage JSON arrays."""
    prompts = [_render(tok, texts) for texts, _ in SMOKE_PROBES]
    outs = llm.generate(prompts, sp)
    ok = True
    for (texts, must), o in zip(SMOKE_PROBES, outs):
        raw = o.outputs[0].text
        arr = parse_array(raw, len(texts))
        prob = []
        if arr is None:
            prob.append("BAD-JSON/len")
        else:
            if any(not x.strip() for x in arr):
                prob.append("empty-string")
            if any(_is_garbage(x) for x in arr):
                prob.append("GARBAGE-repeat")
            joined = " ".join(arr) if arr else ""
            miss = [m for m in must if m not in joined]
            if miss:
                prob.append(f"dropped-identifiers:{miss}")
        status = "ok " if not prob else "FAIL"
        ok = ok and not prob
        logger.info(f"  [{status}] in={texts[0][:40]!r} -> {(arr[0][:50] if arr else raw[:50])!r} "
                    f"{';'.join(prob)}")
    logger.success("SMOKE PASS") if ok else logger.error("SMOKE FAIL")
    return ok


def build_llm(args):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              trust_remote_code=True, seed=args.seed,
              **({"kv_cache_dtype": args.kv_cache_dtype} if args.kv_cache_dtype else {}))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens, seed=args.seed)
    return llm, tok, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targeting", choices=["balanced", "blanket"])
    ap.add_argument("--out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto", help="auto = honor FP8 checkpoint; bfloat16 = fallback")
    ap.add_argument("--kv_cache_dtype", default="", help="e.g. fp8 to shrink KV cache (optional)")
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel GPUs per process")
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="qw", help="id suffix tag (keeps arms separate: -qw{c})")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="run probe gate then exit (0=pass,1=fail)")
    args = ap.parse_args()

    llm, tok, sp = build_llm(args)

    if args.smoke:
        sys.exit(0 if run_smoke(llm, tok, sp) else 1)

    if not (args.targeting and args.out):
        ap.error("--targeting and --out are required unless --smoke")

    # self-gate: even a real shard refuses to generate if its own probes look like garbage.
    if not run_smoke(llm, tok, sp):
        logger.error("shard aborting: smoke gate failed on this process")
        sys.exit(1)

    samples, y, pool = augmentable_pool()
    targets = select_targets(samples, y, pool, args.targeting, seed=args.seed)
    if args.limit:
        targets = targets[:args.limit]
    work = [(idx, c) for idx, k in targets for c in range(k)]
    work = work[args.shard::args.nshards]
    logger.info(f"Qwen {args.targeting} shard {args.shard}/{args.nshards}: {len(work)} copies")

    kept = skipped = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for start in range(0, len(work), args.chunk):
            batch = work[start:start + args.chunk]
            prompts, meta = [], []
            for idx, c in batch:
                s = samples[idx]
                texts = [read_span(s, sp_) for sp_ in get_freetext_spans(s)]
                prompts.append(_render(tok, texts))
                meta.append((idx, c, len(texts)))
            outs = llm.generate(prompts, sp)
            for (idx, c, n), o in zip(meta, outs):
                arr = parse_array(o.outputs[0].text, n)
                if arr is None:
                    skipped += 1
                    continue
                s = samples[idx]
                aug = apply_spans(s, arr)
                fout.write(json.dumps(
                    {"id": f"{s['id']}-{args.suffix}{c}", "session_meta": aug["session_meta"],
                     "history": aug["history"], "current_prompt": aug["current_prompt"],
                     "label": label_of(samples, y, idx)}, ensure_ascii=False) + "\n")
                kept += 1
            fout.flush()
            logger.info(f"  {min(start + args.chunk, len(work))}/{len(work)} -> {kept} kept, "
                        f"{skipped} skipped")
    logger.success(f"Qwen {args.targeting} shard {args.shard}: {kept} rows ({skipped} bad-JSON) "
                   f"-> {args.out}")


if __name__ == "__main__":
    main()
