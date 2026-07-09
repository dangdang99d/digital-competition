"""Data loading, context serialization, split, and the competition Macro-F1 metric.

Shared by the linear-probe experiments. The task: predict an AI coding agent's next
`action` (1 of 14) from session state + history + current prompt.
"""
import csv
import json
import os
import re

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

# confusion groups (92% of val errors stay inside one of these; error analysis)
ACTION_GROUPS = {
    "explore": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "edit": ["edit_file", "write_file", "apply_patch"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "noncode": ["ask_user", "plan_task", "web_search", "respond_only"],
}
GROUP_OF = {a: g for g, acts in ACTION_GROUPS.items() for a in acts}
# group index per class id, aligned with ALL_CLASSES order
GROUP_ID = [list(ACTION_GROUPS).index(GROUP_OF[c]) for c in ALL_CLASSES]


def load_samples(data_dir):
    """Return (samples, y) where samples are the raw JSON dicts and y is a list of action strings."""
    with open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(data_dir, "train_labels.csv"), encoding="utf-8") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    y = [labels[s["id"]] for s in samples]
    return samples, y


# Actions whose result never varies (ask_user: "clarifying question sent to user",
# plan_task: "plan with N steps drafted") — the verdict carries zero information,
# so lean serialization emits them with no "-> ok" suffix at all.
_NO_VERDICT_ACTIONS = ("ask_user", "plan_task")

# Empty search/listing results ("found nothing") count as fail: the verdict means
# "did this advance the task", not "did the tool crash". Leading-count check keeps
# "3 entries (0 files, 3 dirs)" ok while "0 entries (0 files, 0 dirs)" fails.
# "no relevant results" = web_search's empty phrasing (misses the other prefixes).
_EMPTY_RESULT_PREFIXES = ("0 ", "no matches", "empty directory", "found 0 ",
                          "no relevant results")
# lint_or_typecheck reports "N errors, M files affected" (N>=1 in the data);
# finding errors = the check failed, by analogy with "FAIL: N tests failing".
_LINT_ERRORS_RE = re.compile(r"[1-9]\d* errors?,")


def _result_ok(rs):
    """Collapse a result_summary to pass/fail. Fail = ERROR*/FAIL*/exit=nonzero/
    empty search result/lint errors; everything else (ok/PASS/exit=0/counts) = ok."""
    if rs.startswith(("ERROR", "FAIL")):
        return False
    if rs.startswith("exit="):
        return rs.startswith("exit=0")
    if rs.startswith(_EMPTY_RESULT_PREFIXES):
        return False
    if _LINT_ERRORS_RE.match(rs):
        return False
    return True


def _budget_bucket(tokens):
    """names-style budget bucketing (teammate's thresholds, kept identical)."""
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
    """Directory simplification for path-like arg values: keep the last path
    component ('src/a/b.py' -> 'b.py'), preserving a trailing slash marker
    ('configs/' -> 'configs/', './' -> './'). Glob patterns lose their dir part
    too ('**/*.py' -> '*.py') — consistent with the paths-are-harmful hypothesis."""
    if "/" not in v:
        return v
    tail = "/" if v.endswith("/") else ""
    return v.rstrip("/").rsplit("/", 1)[-1] + tail


# Serialization ablation presets: each variant flips exactly ONE axis vs v1.
SERIALIZE_VARIANTS = {
    "v1":        {},                        # the hist0-baseline format
    "nometa":    {"drop_meta": True},       # no [tier=... ci=...] header line
    # ACTION name -> ok|fail: args/summary dropped (verdict rule v2), markers
    # kept — tests content removal alone. Bare lines belong with atomic tokens
    # (combo): without them the "role from leading name token" argument fails.
    "leanact":   {"lean_actions": True},
    "dupprompt": {"dup_prompt": True},      # prompt text doubled inside the PROMPT line
    # bare compressed action lines: "grep_search fail" — no ACTION/-> markers.
    # ALWAYS pair with --special_tokens (atomic names carry the line's role).
    "bareact":   {"lean_actions": True, "bare_actions": True},
    # all axes at once (pair with --special_tokens for the full combo run)
    "combo":     {"drop_meta": True, "lean_actions": True, "dup_prompt": True,
                  "bare_actions": True},
    # names-style metadata enrichment (teammate's headline changes, structure
    # unchanged): basename-only open files + count, bucketed budget/elapsed,
    # dominant code language, loc. Hypothesis under test: full directory paths
    # in open= drag ambiguous explore cases toward read_file.
    "richmeta":  {"rich_meta": True},
    # richmeta + directory simplification inside history action args too
    # (path/scope/pattern values). One axis vs richmeta: v1 -> richmeta -> richargs.
    "richargs":  {"rich_meta": True, "arg_basenames": True},
    # teammate Ki Min Seo's exact SOTA-0.77427 format — a DIFFERENT structure
    # ([META]/[HIST]/[CUR], hist-cap 12); handled specially in build_texts via
    # serialize_names(), NOT serialize(). Empty dict = just a valid --serialize choice.
    "names":     {},
}


def serialize(r, max_hist=None, hist_dropout=0.0, rng=None,
              drop_meta=False, lean_actions=False, dup_prompt=False,
              bare_actions=False, rich_meta=False, arg_basenames=False,
              strip_history=False):
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
    if rich_meta:
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
    elif not drop_meta:
        parts.append(
            f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
            f"budget={sm['budget_tokens_remaining']} ci={ws['last_ci_status']} "
            f"dirty={ws['git_dirty']} open={','.join(ws['open_files']) or '-'}]"
        )
    hist = [] if strip_history else (r["history"] if max_hist is None else r["history"][-max_hist:])
    if hist_dropout and rng is not None:
        hist = [t for t in hist if rng.random() >= hist_dropout]
    for t in hist:
        if t.get("role") == "user":
            parts.append(f"USER: {t['content']}")
        elif lean_actions:
            if t["name"] in _NO_VERDICT_ACTIONS:
                line = t["name"]
            else:
                ok = "ok" if _result_ok(t.get("result_summary", "")) else "fail"
                line = f"{t['name']} {ok}"
            parts.append(line if bare_actions else
                         "ACTION " + line.replace(" ", " -> ", 1))
        else:
            args = t.get("args", {}) or {}
            if arg_basenames:
                args = {k: _strip_dirs(v) if isinstance(v, str) else v
                        for k, v in args.items()}
            parts.append(
                f"ACTION {t['name']}({args}) -> {t.get('result_summary', '')}"
            )
    if dup_prompt:
        parts.append(f"PROMPT: {r['current_prompt']} {r['current_prompt']}")
    else:
        parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


# ---- teammate Ki Min Seo's 'names' serialization (his SOTA-0.77427 granite format) ----
# Ported byte-for-byte from submit_names_single.zip:render_sample. His format is a
# DIFFERENT structure from ours ([META]/[HIST]/[CUR] single line, history capped at 12),
# so it gets its own function + a special branch in build_texts (NOT serialize()). Helpers
# are self-contained (his exact logic, incl. non-int -> "unknown") to guarantee an identical
# string. Used to train granite on his serialization for a clean serialization-only compare.
def _names_safe_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _names_compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _names_budget_bucket(tokens):
    try:
        tokens = int(tokens)
    except (TypeError, ValueError):
        return "unknown"
    if tokens < 2_000:
        return "very_low"
    if tokens < 10_000:
        return "low"
    if tokens < 50_000:
        return "medium"
    return "high"


def _names_elapsed_bucket(seconds):
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if seconds < 120:
        return "early"
    if seconds < 900:
        return "mid"
    return "late"


def serialize_names(r, max_history_events=12):
    """Ki Min Seo's 'names' serialization, byte-identical to his render_sample."""
    meta = r.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = r.get("history") or []
    recent_history = history[-max_history_events:]

    open_files = workspace.get("open_files") or []
    language_mix = workspace.get("language_mix") or {}
    main_lang = ""
    if language_mix:
        main_lang = max(language_mix.items(), key=lambda item: item[1])[0]

    meta_text = " ".join([
        f"tier={_names_safe_text(meta.get('user_tier'))}",
        f"pref={_names_safe_text(meta.get('language_pref'))}",
        f"turn={_names_safe_text(meta.get('turn_index'))}",
        f"budget={_names_budget_bucket(meta.get('budget_tokens_remaining'))}",
        f"elapsed={_names_elapsed_bucket(meta.get('elapsed_session_sec'))}",
        f"lang={main_lang}",
        f"ci={_names_safe_text(workspace.get('last_ci_status'))}",
        f"git={'dirty' if workspace.get('git_dirty') else 'clean'}",
        f"open={len(open_files)}",
        f"loc={_names_safe_text(workspace.get('loc'))}",
    ])
    _names = [f.split("/")[-1] for f in open_files[:6]]
    meta_text = meta_text + " openfiles=" + " ".join(_names)

    hist_parts = []
    for item in recent_history:
        role = item.get("role", "")
        if role == "user":
            hist_parts.append(f"U: {_names_safe_text(item.get('content'))}")
        elif role == "assistant_action":
            name = _names_safe_text(item.get("name"))
            args = _names_compact_json(item.get("args") or {})
            result = _names_safe_text(item.get("result_summary"))
            hist_parts.append(f"A[{name}] {args} -> {result}")

    return " ".join(["[META]", meta_text, "[HIST]", " | ".join(hist_parts),
                     "[CUR]", _names_safe_text(r.get("current_prompt"))])


def build_texts(samples, input_mode="context", max_hist=None, variant="v1",
                strip_history=False):
    """input_mode: 'context' = full serialized; 'prompt' = current_prompt only (baseline-style).
    max_hist=None -> full history. variant: a SERIALIZE_VARIANTS key.
    strip_history: drop all history events (synthesize zero-history samples)."""
    if input_mode == "prompt":
        return [s["current_prompt"] or "" for s in samples]
    if input_mode == "context":
        if variant == "names":
            # his exact serialization; his hist-cap is 12 (used when max_hist unset)
            mh = max_hist if isinstance(max_hist, int) and max_hist > 0 else 12
            return [serialize_names(s, max_history_events=mh) for s in samples]
        kw = SERIALIZE_VARIANTS[variant]
        return [serialize(s, max_hist=max_hist, strip_history=strip_history, **kw)
                for s in samples]
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
