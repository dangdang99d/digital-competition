"""DACON 236694 submission — bge-m3 full-FT + logit-bias calibration.

Layout (mirrors baseline/): model/ + script.py + requirements.txt, zipped.
Reads ./data/test.jsonl, writes ./output/submission.csv. Offline-safe.

The shipped model is VOCAB-PRUNED fp16 (<=1GB limit): the tokenizer is the
complete 250k-piece original, but the embedding matrix keeps only ~55k rows
(tokens used by the data + frequency margin). model/bge-m3/remap.npy maps
tokenizer ids -> pruned rows (pruned ids -> <unk>); it is applied to input_ids
right after collation. Verified prediction-identical to the fp32 full-vocab
model on 3k val samples.

IMPORTANT: serialize() below must stay IDENTICAL to src/data.py serialize()
with max_hist=None — the exact text format the checkpoint was trained on
(bracket meta + full history + PROMPT:, tokenizer truncation at 512).
"""
import csv
import json
import os
from pathlib import Path

# the zip ships exactly one model directory under ./model — auto-detect it
MODEL_DIR = next(p for p in sorted(Path("./model").iterdir()) if p.is_dir())
DATA_DIR = Path("./data")
OUTPUT_PATH = Path("./output/submission.csv")
MAX_LENGTH = 512
BATCH_SIZE = 64


def serialize(r, max_hist=None):
    """Copy of src/data.py serialize() — do not edit independently."""
    sm = r["session_meta"]
    ws = sm["workspace"]
    parts = [
        f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
        f"budget={sm['budget_tokens_remaining']} ci={ws['last_ci_status']} "
        f"dirty={ws['git_dirty']} open={','.join(ws['open_files']) or '-'}]"
    ]
    hist = r["history"] if max_hist is None else r["history"][-max_hist:]
    for t in hist:
        if t.get("role") == "user":
            parts.append(f"USER: {t['content']}")
        else:
            parts.append(
                f"ACTION {t['name']}({t.get('args', {})}) -> {t.get('result_summary', '')}"
            )
    parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_logit_bias(model_dir, id2label):
    path = Path(model_dir) / "logit_bias.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    bias_map = payload.get("bias", {})
    return [float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))]


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # decoder-style tokenizers (Qwen3)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device).eval()
    # tokenizer-id -> pruned-embedding-row map (see module docstring)
    remap = torch.from_numpy(np.load(MODEL_DIR / "remap.npy")).long().to(device)

    id2label = model.config.id2label
    bias_values = load_logit_bias(MODEL_DIR, id2label)
    bias = (torch.tensor(bias_values, dtype=torch.float32, device=device)
            if bias_values is not None else None)

    samples = load_jsonl(DATA_DIR / "test.jsonl")
    ids = [s["id"] for s in samples]
    texts = [serialize(s) for s in samples]

    # tokenize once (no padding), then process in LENGTH-SORTED batches: samples
    # vary a lot in length, and sorting removes padding waste (~1.5-2x faster,
    # which matters against the 10-min inference budget). Predictions are
    # restored to the original order afterwards.
    encodings = []
    for start in range(0, len(texts), 1024):   # batched -> parallel rust tokenizer (3 vCPU)
        enc = tokenizer(texts[start:start + 1024], truncation=True, max_length=MAX_LENGTH, padding=False)
        encodings.extend({"input_ids": i, "attention_mask": a}
                         for i, a in zip(enc["input_ids"], enc["attention_mask"]))
    order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]["input_ids"]), reverse=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    preds_sorted = []
    with torch.no_grad():
        for start in range(0, len(order), BATCH_SIZE):
            chunk = [encodings[i] for i in order[start:start + BATCH_SIZE]]
            batch = {k: v.to(device) for k, v in collator(chunk).items()}
            batch["input_ids"] = remap[batch["input_ids"]]
            logits = model(**batch).logits.float()
            if bias is not None:
                logits = logits + bias
            preds_sorted.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    preds = [None] * len(order)
    for pos, orig_idx in enumerate(order):
        preds[orig_idx] = id2label[int(preds_sorted[pos])]

    pred_map = dict(zip(ids, preds))
    with open(DATA_DIR / "sample_submission.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["action"] = pred_map[row["id"]]

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT_PATH} rows={len(rows)}")


if __name__ == "__main__":
    main()
