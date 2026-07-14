"""DACON action-decision inference — granite E8a+LS+AWP (t031), bitsandbytes-quantized.

Serialization = richargs (byte-for-byte src/data.py serialize, rich_meta+arg_basenames).
Quantization scheme read from quant.json ({"quant": "nf4"} | {"quant": "int8"}); the fp16
weights ship in model/ and are quantized AT LOAD (matches the E46 arm-A measurement exactly).
Length-sorted batching cuts padding waste for the 10-min cap. Raw argmax, NO logit calibration.
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


def _budget_bucket(t):
    if t < 2_000:
        return "very_low"
    if t < 10_000:
        return "low"
    if t < 50_000:
        return "medium"
    return "high"


def _elapsed_bucket(s):
    if s < 120:
        return "early"
    if s < 900:
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
            args = {k: _strip_dirs(v) if isinstance(v, str) else v for k, v in args.items()}
            parts.append(f"ACTION {t['name']}({args}) -> {t.get('result_summary', '')}")
    parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_model_and_tokenizer(model_dir):
    """Load the classifier. If model/ ships PRE-quantized weights (config has a
    quantization_config), load them directly; else fall back to quantize-at-load
    driven by quant.json ({"quant":"nf4"|"int8"})."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.truncation_side = "right"

    cfg_json = json.load(open(os.path.join(model_dir, "config.json")))
    if "quantization_config" in cfg_json:  # pre-quantized weights shipped
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, device_map={"": 0}).eval()
        return model, tokenizer

    quant = json.load(open("quant.json"))["quant"]  # quantize-at-load fallback
    if quant == "nf4":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
    elif quant == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"unknown quant {quant}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, quantization_config=cfg,
        device_map={"": 0}).eval()
    return model, tokenizer


def predict_logits(texts, model, tokenizer, max_length=512, batch_size=128):
    """Length-sorted batched inference; returns logits [N,14] in ORIGINAL order."""
    import numpy as np
    import torch
    # sort by token length to minimize padding
    tok_lens = [len(tokenizer.encode(t, truncation=True, max_length=max_length)) for t in texts]
    order = np.argsort(tok_lens, kind="stable")
    logits = np.empty((len(texts), len(ALL_CLASSES)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = tokenizer([texts[j] for j in idx], truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**batch).logits
            logits[idx] = out.float().cpu().numpy()
    return logits


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np

    data_dir = Path("./data")
    model_root = Path("./model")
    output_path = Path("./output/submission.csv")

    model_dir = sorted(model_root.glob("granite-311m-e8a-ls*"))[0]

    samples = load_jsonl(data_dir / "test.jsonl")
    ids = [s["id"] for s in samples]
    texts = [serialize_richargs(s) for s in samples]

    model, tokenizer = build_model_and_tokenizer(str(model_dir))
    id2label = model.config.id2label
    logits = predict_logits(texts, model, tokenizer)
    preds = [id2label[int(i)] for i in logits.argmax(-1)]

    pred_map = dict(zip(ids, preds))
    with open(data_dir / "sample_submission.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        row["action"] = pred_map[row["id"]]
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {output_path} rows={len(rows)} | model={model_dir.name}")


if __name__ == "__main__":
    main()
