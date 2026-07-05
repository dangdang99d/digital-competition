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


def _result_ok(rs):
    """Collapse a result_summary to pass/fail. Fail = ERROR*/FAIL*/exit=nonzero;
    everything else (ok/PASS/exit=0/counts/plans/questions) = ok."""
    if rs.startswith(("ERROR", "FAIL")):
        return False
    if rs.startswith("exit="):
        return rs.startswith("exit=0")
    return True


# Serialization ablation presets: each variant flips exactly ONE axis vs v1.
SERIALIZE_VARIANTS = {
    "v1":        {},                        # the hist0-baseline format
    "nometa":    {"drop_meta": True},       # no [tier=... ci=...] header line
    "leanact":   {"lean_actions": True},    # ACTION name -> ok|fail (no args/summary)
    "dupprompt": {"dup_prompt": True},      # prompt text doubled inside the PROMPT line
}


def serialize(r, max_hist=None, hist_dropout=0.0, rng=None,
              drop_meta=False, lean_actions=False, dup_prompt=False):
    """Flatten one sample to a single text string: session_meta + history + current_prompt.

    max_hist=None -> use the FULL history (no cap); an int caps to the last N events.
    (Token-level truncation is still handled downstream by the tokenizer's max_len.)
    hist_dropout: training-time augmentation — drop each history event independently
    with this probability (needs rng, a np.random.Generator). Never use for eval.
    drop_meta / lean_actions / dup_prompt: serialization-ablation axes, see
    SERIALIZE_VARIANTS. Defaults reproduce v1 byte-for-byte.
    """
    sm = r["session_meta"]
    ws = sm["workspace"]
    parts = []
    if not drop_meta:
        parts.append(
            f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
            f"budget={sm['budget_tokens_remaining']} ci={ws['last_ci_status']} "
            f"dirty={ws['git_dirty']} open={','.join(ws['open_files']) or '-'}]"
        )
    hist = r["history"] if max_hist is None else r["history"][-max_hist:]
    if hist_dropout and rng is not None:
        hist = [t for t in hist if rng.random() >= hist_dropout]
    for t in hist:
        if t.get("role") == "user":
            parts.append(f"USER: {t['content']}")
        elif lean_actions:
            ok = "ok" if _result_ok(t.get("result_summary", "")) else "fail"
            parts.append(f"ACTION {t['name']} -> {ok}")
        else:
            parts.append(
                f"ACTION {t['name']}({t.get('args', {})}) -> {t.get('result_summary', '')}"
            )
    if dup_prompt:
        parts.append(f"PROMPT: {r['current_prompt']} {r['current_prompt']}")
    else:
        parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


def build_texts(samples, input_mode="context", max_hist=None, variant="v1"):
    """input_mode: 'context' = full serialized; 'prompt' = current_prompt only (baseline-style).
    max_hist=None -> full history. variant: a SERIALIZE_VARIANTS key."""
    if input_mode == "prompt":
        return [s["current_prompt"] or "" for s in samples]
    if input_mode == "context":
        kw = SERIALIZE_VARIANTS[variant]
        return [serialize(s, max_hist=max_hist, **kw) for s in samples]
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
