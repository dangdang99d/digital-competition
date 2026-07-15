"""E39 LLM augmentation via the OpenAI regular API (gpt-4o-mini), async + throttled concurrency.
Generates balanced + blanket for the whole fold-0 pool. Runs locally (no GPU). Incremental,
resumable (skips ids already written), self-throttling (client retries + semaphore).

  python -m experiments.word_aug.gen_llm_api [--concurrency 30] [--model gpt-4o-mini]
"""
import argparse
import asyncio
import json
import os

from experiments.word_aug.common import (apply_spans, augmentable_pool, get_freetext_spans,
                                         label_of, read_span, select_targets)
from experiments.word_aug.gen_llm import SYS, build_prompt, parse_array

OUTDIR = "output/e39/openai_batch"


def load_key():
    for line in open(".env"):
        if line.startswith("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
            return


async def main_async(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(max_retries=8)          # built-in exp-backoff on 429/5xx
    os.makedirs(OUTDIR, exist_ok=True)

    samples, y, pool = augmentable_pool()
    work = []
    for tgt in ("balanced", "blanket"):
        for idx, k in select_targets(samples, y, pool, tgt, seed=42):
            for c in range(k):
                work.append((tgt, int(idx), c))

    # resume: skip ids already written — PER TARGETING (id doesn't encode targeting, so a
    # balanced and blanket row for the same (idx,copy) share an id; must not cross-skip).
    outp = {t: f"{OUTDIR}/llm_{t}.jsonl" for t in ("balanced", "blanket")}
    done = {t: set() for t in ("balanced", "blanket")}
    for t, p in outp.items():
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    done[t].add(json.loads(line)["id"])
    ndone = sum(len(s) for s in done.values())
    work = [(t, i, c) for (t, i, c) in work if f"{samples[i]['id']}-llm{c}" not in done[t]]
    print(f"work: {len(work)} rows to generate ({ndone} already done)", flush=True)

    outs = {t: open(outp[t], "a", encoding="utf-8") for t in ("balanced", "blanket")}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    ctr = {"balanced": 0, "blanket": 0, "skipped": 0, "error": 0, "in_tok": 0, "out_tok": 0}
    q = asyncio.Queue()
    for w in work:
        q.put_nowait(w)

    import traceback

    async def one(tgt, idx, c):
        """Process one row. ALL exceptions caught here so a single failure can't kill the loop."""
        try:
            s = samples[idx]
            texts = [read_span(s, sp_) for sp_ in get_freetext_spans(s)]
            async with sem:
                resp = await client.chat.completions.create(
                    model=args.model, temperature=0.7, max_tokens=512,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": build_prompt(texts)}])
            ctr["in_tok"] += resp.usage.prompt_tokens
            ctr["out_tok"] += resp.usage.completion_tokens
            arr = parse_array(resp.choices[0].message.content, len(texts))
            if arr is None:
                ctr["skipped"] += 1
                return
            aug = apply_spans(s, arr)
            row = {"id": f"{s['id']}-llm{c}", "session_meta": aug["session_meta"],
                   "history": aug["history"], "current_prompt": aug["current_prompt"],
                   "label": label_of(samples, y, idx)}
            async with lock:
                outs[tgt].write(json.dumps(row, ensure_ascii=False) + "\n")
                outs[tgt].flush()
            ctr[tgt] += 1
        except Exception:
            ctr["error"] += 1
            if ctr["error"] <= 5:
                print("WORKER ERROR:\n" + traceback.format_exc(), flush=True)

    async def worker():
        while True:
            try:
                tgt, idx, c = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            await one(tgt, idx, c)
            n = ctr["balanced"] + ctr["blanket"] + ctr["skipped"] + ctr["error"]
            if n % 1000 == 0:
                cost = ctr["in_tok"] * 0.15 / 1e6 + ctr["out_tok"] * 0.60 / 1e6
                print(f"  {n}/{len(work)} | bal={ctr['balanced']} bln={ctr['blanket']} "
                      f"skip={ctr['skipped']} err={ctr['error']} | ${cost:.2f}", flush=True)
            q.task_done()

    try:
        await asyncio.gather(*[worker() for _ in range(args.concurrency)])
    except Exception:
        print("GATHER CRASHED:\n" + traceback.format_exc(), flush=True)
    for f in outs.values():
        f.close()
    cost = ctr["in_tok"] * 0.15 / 1e6 + ctr["out_tok"] * 0.60 / 1e6
    print(f"DONE bal={ctr['balanced']} bln={ctr['blanket']} skipped={ctr['skipped']} "
          f"err={ctr['error']} | tokens in={ctr['in_tok']} out={ctr['out_tok']} | ${cost:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    load_key()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
