"""DACON action-decision inference — E26 two-member granite ensemble.

Zip layout:
  model/tokenizer/            one COMPLETE tokenizer (granite members share it)
  model/remap.npy             tokenizer id -> pruned-embedding row (keep-set = union over
                              all member serializations, so one remap serves every member)
  model/member_<i>_<tag>/     vocab-pruned fp16 granite + serialize_variant.json
                              ({"variant": "richargs"|"richmeta"} — HOW that member's
                              training serialized inputs; richmeta = richargs minus
                              arg-basename stripping)

Pipeline: per DISTINCT variant, serialize + tokenize ONCE; length-sorted batches (bs 256
fp16); members run sequentially (peak VRAM = one model) on their own variant's encodings;
UNIFORM MEAN of member softmax probabilities; raw argmax (NO calibration); restore order.
"""
import csv
import json
import os
from pathlib import Path

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
MAX_LENGTH = 512
BATCH_SIZE = int(os.environ.get("ENS_BS", "256"))


# ----- rich serialization (byte-for-byte from src/data.py:serialize, rich_meta=True;
#       strip_args toggles arg_basenames: True -> richargs, False -> richmeta) -----
def _budget_bucket(tokens):
    if tokens < 2_000:
        return "very_low"
    if tokens < 10_000:
        return "low"
    if tokens < 50_000:
        return "medium"
    return "high"


def _elapsed_bucket(seconds):
    if seconds < 120:
        return "early"
    if seconds < 900:
        return "mid"
    return "late"


def _strip_dirs(v):
    if "/" not in v:
        return v
    tail = "/" if v.endswith("/") else ""
    return v.rstrip("/").rsplit("/", 1)[-1] + tail


def serialize_rich(r, strip_args):
    sm = r["session_meta"]
    ws = sm["workspace"]
    parts = []
    mix = ws.get("language_mix") or {}
    codelang = max(mix.items(), key=lambda kv: kv[1])[0] if mix else "-"
    names = [f.rsplit("/", 1)[-1] for f in ws["open_files"][:6]]
    parts.append(
        f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
        f"budget={_budget_bucket(sm['budget_tokens_remaining'])} "
        f"elapsed={_elapsed_bucket(sm['elapsed_session_sec'])} "
        f"codelang={codelang} loc={ws.get('loc', '-')} ci={ws['last_ci_status']} "
        f"dirty={ws['git_dirty']} open={len(ws['open_files'])} "
        f"files={','.join(names) or '-'}]"
    )
    for t in r["history"]:
        if t.get("role") == "user":
            parts.append(f"USER: {t['content']}")
        else:
            args = t.get("args", {}) or {}
            if strip_args:
                args = {k: _strip_dirs(v) if isinstance(v, str) else v
                        for k, v in args.items()}
            parts.append(f"ACTION {t['name']}({args}) -> {t.get('result_summary', '')}")
    parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


VARIANT_STRIP = {"richargs": True, "richmeta": False}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding)

    data_dir = Path("./data")
    model_root = Path("./model")
    output_path = Path("./output/submission.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    member_dirs = sorted(model_root.glob("member_*"))
    assert member_dirs, "no model/member_* dirs in the zip"
    variants = {}
    for mdir in member_dirs:
        v = json.load(open(mdir / "serialize_variant.json"))["variant"]
        assert v in VARIANT_STRIP, f"unknown variant {v}"
        variants[mdir] = v

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    tokenizer.truncation_side = "right"
    remap = torch.from_numpy(np.load(model_root / "remap.npy")).long().to(device)
    collator = DataCollatorWithPadding(tokenizer)

    samples = load_jsonl(data_dir / "test.jsonl")
    ids = [s["id"] for s in samples]

    # serialize + tokenize ONCE per distinct variant; shared length-sort order per variant
    enc_by_variant, order_by_variant = {}, {}
    for v in set(variants.values()):
        texts = [serialize_rich(s, VARIANT_STRIP[v]) for s in samples]
        e = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
        e = [{"input_ids": e["input_ids"][i], "attention_mask": e["attention_mask"][i]}
             for i in range(len(texts))]
        enc_by_variant[v] = e
        order_by_variant[v] = sorted(range(len(e)), key=lambda i: len(e[i]["input_ids"]))

    mean_probs = np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float64)
    for mdir in member_dirs:
        v = variants[mdir]
        enc, order = enc_by_variant[v], order_by_variant[v]
        model = AutoModelForSequenceClassification.from_pretrained(
            mdir, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
        with torch.no_grad():
            for start in range(0, len(order), BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]
                batch = {k: t.to(device) for k, t in collator([enc[i] for i in idx]).items()}
                batch["input_ids"] = remap[batch["input_ids"]]
                logits = model(**batch).logits.float()
                mean_probs[idx] += torch.softmax(logits, -1).cpu().numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"member done: {mdir.name} ({v})")
    mean_probs /= len(member_dirs)

    # NO calibration: raw argmax of the uniform member-probability mean.
    preds = [ALL_CLASSES[int(i)] for i in mean_probs.argmax(1)]
    pred_map = dict(zip(ids, preds))
    with open(data_dir / "sample_submission.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["action"] = pred_map[row["id"]]
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {output_path} rows={len(rows)} | members={[d.name for d in member_dirs]} "
          f"| bs={BATCH_SIZE}")


if __name__ == "__main__":
    main()
