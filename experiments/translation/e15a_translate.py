"""E15a — NLLB-600M translator qualification for the DACON action-decision data.

Selection study (not training). We:
  1. Sample real Korean-bearing records from data/train.jsonl.
  2. Collect the natural-language fields that a translation pipeline would touch
     (history USER contents + current_prompt) that actually contain Korean.
  3. Translate each KO->EN with facebook/nllb-200-distilled-600M (fp16, GPU 0).
  4. Measure: throughput (sent/s), code-span preservation (verbatim survival of
     paths / identifiers / dotted names / HTTP codes), and dump (orig->trans) pairs.
  5. Optionally score CometKiwi (reference-free QE) if unbabel-comet is importable.

No git commit. Writes JSON + CSV artifacts into experiments/translation/.
"""
import json, re, time, os, sys, argparse

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

KO_RE = re.compile(r"[가-힣]")

# "must be verbatim" code spans embedded in the natural-language text.
# Each pattern targets tokens that are unambiguously code (contain /, ., _,
# CamelCase, ALLCAPS, or are HTTP-status-like numbers) — a translator that
# alters any of these has mangled load-bearing structure. ASCII look-arounds
# are used instead of \b because Korean particles (e.g. "302로", "REST를") are
# \w characters and would otherwise suppress a boundary.
SPAN_PATTERNS = [
    re.compile(r"(?:[A-Za-z0-9_]+/)+[A-Za-z0-9_.\*]+"),               # paths a/b/c.py
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_.]*"),  # dotted / file.ext
    re.compile(r"(?<![A-Za-z])[a-z]+(?:[A-Z][a-z0-9]*)+"),            # camelCase
    re.compile(r"(?<![A-Za-z])[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"),    # PascalCase
    re.compile(r"(?<![A-Za-z0-9_])[a-z0-9]+_[a-z0-9_]+"),             # snake_case
    re.compile(r"(?<![A-Za-z0-9_])[A-Z]{2,}(?:_[A-Z0-9]+)*(?![A-Za-z0-9_])"),  # ALLCAPS / CONST
    re.compile(r"`[^`]+`"),                                           # backtick spans
    re.compile(r"(?<![A-Za-z0-9])[1-5]\d{2}(?![A-Za-z0-9])"),         # HTTP status codes
]


def extract_spans(text):
    raw = set()
    for pat in SPAN_PATTERNS:
        for m in pat.finditer(text):
            s = m.group(0).strip("`")
            if len(s) < 2:
                continue
            raw.add(s)
    # keep only maximal spans: drop any span that is a substring of another
    # captured span in the same text (removes CamelCase sub-fragments and
    # path/file-basename redundancy, so each code entity counts once).
    maximal = {s for s in raw if not any(s != o and s in o for o in raw)}
    return maximal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="records to sample")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="experiments/translation")
    args = ap.parse_args()

    dev = "cuda"
    name = "facebook/nllb-200-distilled-600M"
    print("loading", name, flush=True)
    tok = AutoTokenizer.from_pretrained(name, src_lang="kor_Hang")
    model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    bos = tok.convert_tokens_to_ids("eng_Latn")

    # ---- collect korean-bearing natural-language strings from N records ----
    records = []
    with open("data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if KO_RE.search(o.get("current_prompt", "")) or any(
                t.get("role") == "user" and KO_RE.search(t.get("content", ""))
                for t in o["history"]
            ):
                records.append(o)
            if len(records) >= args.n:
                break

    strings = []   # list of dicts: {rec_id, field, text}
    for o in records:
        for t in o["history"]:
            if t.get("role") == "user" and KO_RE.search(t.get("content", "")):
                strings.append({"id": o["id"], "field": "history_user", "text": t["content"]})
        cp = o.get("current_prompt", "")
        if KO_RE.search(cp):
            strings.append({"id": o["id"], "field": "current_prompt", "text": cp})

    print(f"{len(records)} records -> {len(strings)} korean strings to translate", flush=True)

    # ---- translate in batches, timed ----
    src_texts = [s["text"] for s in strings]
    translations = []
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(src_texts), args.batch):
            batch = src_texts[i:i + args.batch]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(dev)
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 max_new_tokens=args.max_new, num_beams=1)
            translations.extend(tok.batch_decode(gen, skip_special_tokens=True))
    torch.cuda.synchronize()
    dt = time.time() - t0
    sent_per_s = len(src_texts) / dt

    # ---- code-span preservation ----
    total_spans = 0
    kept_spans = 0
    span_fail_examples = []
    for s, tr in zip(strings, translations):
        spans = extract_spans(s["text"])
        for sp in spans:
            total_spans += 1
            if sp in tr:
                kept_spans += 1
            elif len(span_fail_examples) < 40:
                span_fail_examples.append({"id": s["id"], "span": sp,
                                           "src": s["text"], "trans": tr})
    preservation = kept_spans / total_spans if total_spans else float("nan")

    # per-string preservation (fraction of strings with 0 mangled spans)
    clean_strings = 0
    strings_with_spans = 0
    for s, tr in zip(strings, translations):
        spans = extract_spans(s["text"])
        if not spans:
            continue
        strings_with_spans += 1
        if all(sp in tr for sp in spans):
            clean_strings += 1
    frac_clean = clean_strings / strings_with_spans if strings_with_spans else float("nan")

    # ---- write artifacts ----
    os.makedirs(args.out, exist_ok=True)
    pairs = [{"id": s["id"], "field": s["field"], "orig": s["text"], "trans": tr}
             for s, tr in zip(strings, translations)]
    with open(os.path.join(args.out, "e15a_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)

    metrics = {
        "n_records": len(records),
        "n_strings": len(strings),
        "throughput_sent_per_s": sent_per_s,
        "wall_sec": dt,
        "code_span_total": total_spans,
        "code_span_kept": kept_spans,
        "code_span_preservation": preservation,
        "strings_with_spans": strings_with_spans,
        "strings_fully_clean": clean_strings,
        "frac_strings_clean": frac_clean,
    }
    with open(os.path.join(args.out, "e15a_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "span_fail_examples": span_fail_examples},
                  f, ensure_ascii=False, indent=1)

    print("\n==== E15a METRICS ====", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}", flush=True)
    print("\n---- span-fail examples (first 8) ----", flush=True)
    for ex in span_fail_examples[:8]:
        print(f"  MANGLED [{ex['span']}]", flush=True)
        print(f"     src : {ex['src'][:160]}", flush=True)
        print(f"     tran: {ex['trans'][:160]}", flush=True)


if __name__ == "__main__":
    main()
