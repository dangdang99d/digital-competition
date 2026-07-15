"""E39 LLM augmentation via the OpenAI Batch API (gpt-4o-mini, 50% off, async).
Generates balanced + blanket paraphrase sets for the WHOLE fold-0 pool. Runs locally (no GPU).

  submit : build request JSONLs (<=50k/batch), upload, create batches, save batch ids
  status : print each batch's status + counts
  fetch  : download completed outputs, parse -> output/e39/openai_batch/llm_{balanced,blanket}.jsonl

Reads OPENAI_API_KEY from .env. custom_id = "{targeting}|{idx}|{copy}".
"""
import argparse
import json
import os

from experiments.word_aug.common import (apply_spans, augmentable_pool, get_freetext_spans,
                                         label_of, read_span, select_targets)
from experiments.word_aug.gen_llm import SYS, build_prompt, parse_array

MODEL = "gpt-4o-mini"
MAX_PER_BATCH = 6000            # ~200 in-tok/req → ~1.2M tokens, under the 2M enqueued limit
OUTDIR = "output/e39/openai_batch"


def load_key():
    for line in open(".env"):
        if line.startswith("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
            return
    raise SystemExit("OPENAI_API_KEY not in .env")


def build_requests(samples, y, pool):
    reqs = []
    for tgt in ["balanced", "blanket"]:
        targets = select_targets(samples, y, pool, tgt, seed=42)
        work = [(idx, c) for idx, k in targets for c in range(k)]
        for idx, c in work:
            s = samples[idx]
            texts = [read_span(s, sp_) for sp_ in get_freetext_spans(s)]
            reqs.append({
                "custom_id": f"{tgt}|{idx}|{c}",
                "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": MODEL, "temperature": 0.7, "max_tokens": 512,
                         "messages": [{"role": "system", "content": SYS},
                                      {"role": "user", "content": build_prompt(texts)}]}})
    return reqs


def cmd_submit(client):
    samples, y, pool = augmentable_pool()
    reqs = build_requests(samples, y, pool)
    os.makedirs(OUTDIR, exist_ok=True)
    ids = []
    for bi in range(0, len(reqs), MAX_PER_BATCH):
        chunk = reqs[bi:bi + MAX_PER_BATCH]
        fp = f"{OUTDIR}/batch_in_{bi // MAX_PER_BATCH}.jsonl"
        with open(fp, "w", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        up = client.files.create(file=open(fp, "rb"), purpose="batch")
        b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
        ids.append(b.id)
        print(f"batch {bi // MAX_PER_BATCH}: {len(chunk)} reqs -> {b.id} [{b.status}]", flush=True)
    json.dump(ids, open(f"{OUTDIR}/batch_ids.json", "w"))
    print(f"SUBMITTED {len(ids)} batches, {len(reqs)} requests total")


def cmd_status(client):
    ids = json.load(open(f"{OUTDIR}/batch_ids.json"))
    alldone = True
    for bid in ids:
        b = client.batches.retrieve(bid)
        rc = b.request_counts
        print(f"{bid}: {b.status}  ({rc.completed}/{rc.total} done, {rc.failed} failed)")
        if b.status not in ("completed",):
            alldone = False
    print("ALL_COMPLETED" if alldone else "PENDING")


def cmd_fetch(client):
    samples, y, pool = augmentable_pool()
    ids = json.load(open(f"{OUTDIR}/batch_ids.json"))
    outs = {t: open(f"{OUTDIR}/llm_{t}.jsonl", "w", encoding="utf-8") for t in ("balanced", "blanket")}
    kept = {"balanced": 0, "blanket": 0}
    skipped = 0
    for bid in ids:
        b = client.batches.retrieve(bid)
        if b.status != "completed":
            print(f"{bid}: {b.status} — skipping (run fetch again when completed)")
            continue
        text = client.files.content(b.output_file_id).text
        for line in text.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tgt, idx, c = r["custom_id"].split("|")
            idx = int(idx)
            s = samples[idx]
            n = len(get_freetext_spans(s))
            try:
                content = r["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:
                skipped += 1
                continue
            arr = parse_array(content, n)
            if arr is None:
                skipped += 1
                continue
            aug = apply_spans(s, arr)
            outs[tgt].write(json.dumps(
                {"id": f"{s['id']}-llm{c}", "session_meta": aug["session_meta"],
                 "history": aug["history"], "current_prompt": aug["current_prompt"],
                 "label": label_of(samples, y, idx)}, ensure_ascii=False) + "\n")
            kept[tgt] += 1
    for f in outs.values():
        f.close()
    print(f"FETCHED balanced={kept['balanced']} blanket={kept['blanket']} skipped(bad-JSON)={skipped}")


def _parse_output(client, out_file_id, samples, y, outs, ctr):
    text = client.files.content(out_file_id).text
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tgt, idx, c = r["custom_id"].split("|")
        idx = int(idx)
        s = samples[idx]
        n = len(get_freetext_spans(s))
        try:
            content = r["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            ctr["skipped"] += 1
            continue
        arr = parse_array(content, n)
        if arr is None:
            ctr["skipped"] += 1
            continue
        aug = apply_spans(s, arr)
        outs[tgt].write(json.dumps(
            {"id": f"{s['id']}-llm{c}", "session_meta": aug["session_meta"],
             "history": aug["history"], "current_prompt": aug["current_prompt"],
             "label": label_of(samples, y, idx)}, ensure_ascii=False) + "\n")
        outs[tgt].flush()
        ctr[tgt] += 1


def cmd_pipeline(client):
    """Submit ≤MAX_PER_BATCH-request chunks ONE AT A TIME (stays under the 2M enqueued-token
    limit): submit → poll → fetch+append → next. Resumable (pipeline_done.json), incremental."""
    import time
    samples, y, pool = augmentable_pool()
    reqs = build_requests(samples, y, pool)
    os.makedirs(OUTDIR, exist_ok=True)
    nch = (len(reqs) + MAX_PER_BATCH - 1) // MAX_PER_BATCH
    dp = f"{OUTDIR}/pipeline_done.json"
    done = set(json.load(open(dp))) if os.path.exists(dp) else set()
    mode = "a" if done else "w"
    outs = {t: open(f"{OUTDIR}/llm_{t}.jsonl", mode, encoding="utf-8") for t in ("balanced", "blanket")}
    ctr = {"balanced": 0, "blanket": 0, "skipped": 0}
    print(f"pipeline: {len(reqs)} requests in {nch} chunks of {MAX_PER_BATCH} ({len(done)} already done)", flush=True)
    for ci in range(nch):
        if ci in done:
            continue
        chunk = reqs[ci * MAX_PER_BATCH:(ci + 1) * MAX_PER_BATCH]
        fp = f"{OUTDIR}/chunk_{ci}.jsonl"
        with open(fp, "w", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok = False
        for attempt in range(3):
            up = client.files.create(file=open(fp, "rb"), purpose="batch")
            b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                      completion_window="24h")
            while b.status not in ("completed", "failed", "expired", "cancelled"):
                time.sleep(30)
                b = client.batches.retrieve(b.id)
            if b.status == "completed":
                _parse_output(client, b.output_file_id, samples, y, outs, ctr)
                ok = True
                break
            print(f"  chunk {ci} attempt {attempt}: {b.status} {b.errors}; retry in 30s", flush=True)
            time.sleep(30)
        if ok:
            done.add(ci)
            json.dump(sorted(done), open(dp, "w"))
        print(f"chunk {ci + 1}/{nch} {'OK' if ok else 'FAILED'} | "
              f"bal={ctr['balanced']} bln={ctr['blanket']} skip={ctr['skipped']}", flush=True)
    for f in outs.values():
        f.close()
    print(f"PIPELINE DONE bal={ctr['balanced']} bln={ctr['blanket']} skipped={ctr['skipped']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["submit", "status", "fetch", "pipeline"])
    args = ap.parse_args()
    load_key()
    from openai import OpenAI
    client = OpenAI()
    {"submit": cmd_submit, "status": cmd_status, "fetch": cmd_fetch,
     "pipeline": cmd_pipeline}[args.phase](client)


if __name__ == "__main__":
    main()
