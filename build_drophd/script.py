"""DACON action-decision inference (E8a+LS granite, richargs serialization).

Single granite-embedding-311m ModernBert classifier, full-vocab tokenizer.
Serialization MUST match training (variant=richargs). No logit calibration:
raw argmax over the 14 action classes.
"""
import csv
import json
import os
from pathlib import Path

# 14 target classes, fixed order (matches the model's id2label).
ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]


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
    """Keep the last path component, preserving a trailing-slash marker."""
    if "/" not in v:
        return v
    tail = "/" if v.endswith("/") else ""
    return v.rstrip("/").rsplit("/", 1)[-1] + tail


def serialize_richargs(r):
    """Flatten one sample to a single text string (richargs variant)."""
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
    for t in r["history"]:  # full history (max_hist=None at training)
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


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data_dir = Path("./data")
    model_root = Path("./model")
    output_path = Path("./output/submission.csv")
    max_length = 512
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dirs = sorted(model_root.glob("granite-311m-e8a-ls*"))
    if not model_dirs:
        raise FileNotFoundError("no model dir under ./model matching granite-311m-e8a-ls*")
    model_dir = model_dirs[0]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.truncation_side = "right"  # match training

    samples = load_jsonl(data_dir / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = [serialize_richargs(sample) for sample in samples]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True).to(device).eval()
    id2label = model.config.id2label

    logits_all = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], truncation=True,
                              max_length=max_length, padding=True,
                              return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits
            logits_all.append(logits.float().cpu().numpy())
    logits_all = np.concatenate(logits_all, 0)

    # NO logit-bias calibration: raw argmax.
    pred_ids = logits_all.argmax(axis=-1).tolist()
    preds = [id2label[int(i)] for i in pred_ids]

    pred_map = dict(zip(ids, preds))
    fieldnames, rows = load_sample_submission(data_dir / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(output_path, fieldnames, rows)
    print(f"Saved {output_path} rows={len(rows)} | model={model_dir.name}")


if __name__ == "__main__":
    main()
