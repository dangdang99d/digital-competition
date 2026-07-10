"""DACON action-decision inference — E26 two-member granite ensemble (richargs).

Ships as `script.py`. Layout inside the zip:
  model/tokenizer/            one COMPLETE tokenizer (shared; granite members are same backbone)
  model/remap.npy             tokenizer id -> pruned-embedding row (shared: members pruned
                              with the SAME keep-set, so one remap serves both)
  model/member_0_<tag>/       vocab-pruned fp16 granite classifier
  model/member_1_<tag>/       vocab-pruned fp16 granite classifier

Pipeline: serialize richargs (byte-copy of training) -> tokenize ONCE -> length-sorted
batches (bs 256, fp16) -> per member: forward all batches, softmax -> UNIFORM MEAN of member
probabilities -> raw argmax (NO calibration) -> restore order -> submission.csv.
Members run sequentially (load -> forward -> free) so peak VRAM = one model.
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
BATCH_SIZE = int(os.environ.get("ENS_BS", "256"))   # T4-sized: peak VRAM measured on the sweep


# ----- richargs serialization (byte-for-byte copy of src/data.py:serialize
#       with rich_meta=True, arg_basenames=True; used at training time) -----
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


def serialize_richargs(r):
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
            args = {k: _strip_dirs(v) if isinstance(v, str) else v
                    for k, v in args.items()}
            parts.append(f"ACTION {t['name']}({args}) -> {t.get('result_summary', '')}")
    parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


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
    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    tokenizer.truncation_side = "right"
    remap = torch.from_numpy(np.load(model_root / "remap.npy")).long().to(device)
    collator = DataCollatorWithPadding(tokenizer)

    samples = load_jsonl(data_dir / "test.jsonl")
    ids = [s["id"] for s in samples]
    texts = [serialize_richargs(s) for s in samples]

    # tokenize ONCE (shared tokenizer), then length-sort to kill padding waste
    encodings = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
    encodings = [{"input_ids": encodings["input_ids"][i],
                  "attention_mask": encodings["attention_mask"][i]}
                 for i in range(len(texts))]
    order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]["input_ids"]))

    mean_probs = None
    for mdir in member_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(
            mdir, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
        probs_sorted = []
        with torch.no_grad():
            for start in range(0, len(order), BATCH_SIZE):
                chunk = [encodings[i] for i in order[start:start + BATCH_SIZE]]
                batch = {k: v.to(device) for k, v in collator(chunk).items()}
                batch["input_ids"] = remap[batch["input_ids"]]
                logits = model(**batch).logits.float()
                probs_sorted.append(torch.softmax(logits, -1).cpu().numpy())
        probs_sorted = np.concatenate(probs_sorted, 0)
        mean_probs = probs_sorted if mean_probs is None else mean_probs + probs_sorted
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"member done: {mdir.name}")
    mean_probs /= len(member_dirs)

    # NO calibration: raw argmax of the uniform member-probability mean.
    id2label = {i: c for i, c in enumerate(ALL_CLASSES)}
    preds_sorted = mean_probs.argmax(1)
    preds = [None] * len(order)
    for pos, orig_idx in enumerate(order):
        preds[orig_idx] = id2label[int(preds_sorted[pos])]

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
