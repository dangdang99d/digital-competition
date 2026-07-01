"""Data loading, context serialization, split, and the competition Macro-F1 metric.

Shared by the linear-probe experiments. The task: predict an AI coding agent's next
`action` (1 of 14) from session state + history + current prompt.
"""
import csv
import json
import os

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 14 target classes (fixed order — used for the Macro-F1 label set)
ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
CLASS_TO_ID = {c: i for i, c in enumerate(ALL_CLASSES)}


def load_samples(data_dir):
    """Return (samples, y) where samples are the raw JSON dicts and y is a list of action strings."""
    with open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(data_dir, "train_labels.csv"), encoding="utf-8") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    y = [labels[s["id"]] for s in samples]
    return samples, y


def serialize(r, max_hist=6):
    """Flatten one sample to a single text string: session_meta + recent history + current_prompt.

    Matches analysis/llm_clustering.ipynb so probe inputs are consistent with the EDA.
    """
    sm = r["session_meta"]
    ws = sm["workspace"]
    parts = [
        f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
        f"budget={sm['budget_tokens_remaining']} ci={ws['last_ci_status']} "
        f"dirty={ws['git_dirty']} open={','.join(ws['open_files']) or '-'}]"
    ]
    for t in r["history"][-max_hist:]:
        if t.get("role") == "user":
            parts.append(f"USER: {t['content']}")
        else:
            parts.append(
                f"ACTION {t['name']}({t.get('args', {})}) -> {t.get('result_summary', '')}"
            )
    parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


def build_texts(samples, input_mode="context", max_hist=6):
    """input_mode: 'context' = full serialized; 'prompt' = current_prompt only (baseline-style)."""
    if input_mode == "prompt":
        return [s["current_prompt"] or "" for s in samples]
    if input_mode == "context":
        return [serialize(s, max_hist=max_hist) for s in samples]
    raise ValueError(f"unknown input_mode: {input_mode}")


def split_indices(y, test_size=0.2, seed=42):
    """Stratified train/val split over row indices."""
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return tr, va


def macro_f1(y_true_ids, y_pred_ids):
    """Competition metric: mean of per-class binary F1 over all 14 classes.

    y_true_ids / y_pred_ids are integer class ids in [0, 14). Uses the fixed 14-label
    set so absent classes still count (zero_division=0), matching the DACON scorer.
    """
    return f1_score(
        y_true_ids, y_pred_ids,
        labels=list(range(len(ALL_CLASSES))),
        average="macro", zero_division=0,
    )
