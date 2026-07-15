"""E39 word-level augmentation — shared foundation for both generators (C-MLM, LLM).

Free-text-only, leak-safe, code-identifier-aware. Everything here is offline data prep;
only the trained classifier ships. src/ untouched.

Augmentable pool = the full_data TRAIN rows (tr ∪ va_train) — NEVER the 3.5k held-out
va_eval (finetune.py:1241), so generated rows can't leak into eval. Two targetings:
  - rare   : upsample the below-median classes toward the median (macro-F1 play)
  - blanket: one paraphrase per training row
Augmented rows are written as JSONL with the SAME sample schema + a "label" (action string)
and a fresh id ("<origid>-augN"); the E39 train loader unions them with the originals.
"""
import json
import re

import numpy as np

from src.data import load_samples, session_fold_indices


def augmentable_pool(data_dir="./data", fold=0, n_splits=5, seed=42):
    """Return (samples, y, pool_idx). E39 baseline = the AWP-Optuna **t070** recipe, whose
    A0 control (fold-F1 0.7806) is already trained on **session-fold-0** (leak-free
    StratifiedGroupKFold by session — E38). pool = that fold's 56k TRAIN rows only; the ~14k
    holdout is where we eval, so augmentation (sourced only from train) can never leak."""
    samples, y = load_samples(data_dir)
    tr, va = session_fold_indices(samples, y, fold, n_splits=n_splits, seed=seed)
    return samples, y, np.sort(tr)


# ---- code-identifier protection (papers' named-entity failure mode) ----
_CAMEL = re.compile(r"[a-z][A-Z]")
_PATHY = re.compile(r"[/\\.(){}\[\]<>=:@#]|::")
def is_code_token(w):
    """True if a whitespace token looks like code/identifier/path/number → do NOT alter."""
    core = w.strip("`'\".,!?);:")
    if not core:
        return False
    if any(ch.isdigit() for ch in core):
        return True
    if "_" in core or _CAMEL.search(core) or _PATHY.search(w):
        return True
    if core.isupper() and len(core) > 1:               # ALLCAPS constants (REST, CI)
        return True
    if "`" in w:                                       # backtick-wrapped
        return True
    return False


# ---- free-text spans of a sample (current_prompt + USER contents) ----
def get_freetext_spans(sample):
    """List of (kind, key) locators for the editable free-text of a sample, in order."""
    spans = [("prompt", None)]
    for i, t in enumerate(sample.get("history", [])):
        if t.get("role") == "user" and t.get("content"):
            spans.append(("user", i))
    return spans


def read_span(sample, span):
    kind, i = span
    return sample["current_prompt"] if kind == "prompt" else sample["history"][i]["content"]


def apply_spans(sample, span_texts):
    """Return a shallow-copied sample with free-text spans replaced by span_texts (same order
    as get_freetext_spans). Non-user history events reused verbatim; structure untouched."""
    spans = get_freetext_spans(sample)
    assert len(spans) == len(span_texts), "span/text count mismatch"
    s2 = dict(sample)
    new_hist = list(sample.get("history", []))
    for (kind, i), txt in zip(spans, span_texts):
        if kind == "prompt":
            s2["current_prompt"] = txt
        else:
            new_hist[i] = dict(new_hist[i])
            new_hist[i]["content"] = txt
    s2["history"] = new_hist
    return s2


# ---- targeting: which pool rows to augment, and how many copies each ----
def _balanced_alloc(y, pool, seed):
    """Copies per source row to lift EVERY class up to the max class count (class balance →
    macro-F1). Sampling with replacement, so a rare row may get several distinct paraphrases."""
    from collections import Counter, defaultdict
    counts = Counter(y[i] for i in pool)
    mx = max(counts.values())
    rng = np.random.default_rng(seed)
    alloc = defaultdict(int)
    for cls, n in counts.items():
        need = mx - n
        if need <= 0:
            continue
        cls_pool = [int(i) for i in pool if y[i] == cls]
        for idx in rng.choice(cls_pool, size=need, replace=True):
            alloc[int(idx)] += 1
    return alloc


def balanced_total(y, pool, seed=42):
    return sum(_balanced_alloc(y, pool, seed).values())


def select_targets(samples, y, pool, targeting, seed=42, n_total=None):
    """Return [(pool_index, n_copies)]. Both configs generate the SAME volume (user: 69k+69k)
    so balanced-vs-blanket isolates TARGETING, not volume.
      balanced : lift every class up to the max count (class balance).
      blanket  : n_total copies sampled UNIFORMLY from the pool; n_total defaults to the
                 balanced volume so the two arms match exactly."""
    if targeting == "balanced":
        return list(_balanced_alloc(y, pool, seed).items())
    if targeting == "blanket":
        if n_total is None:
            n_total = balanced_total(y, pool, seed)    # auto-match balanced volume
        from collections import Counter
        rng = np.random.default_rng(seed + 1)
        picks = rng.choice([int(i) for i in pool], size=n_total, replace=True)
        return [(idx, int(k)) for idx, k in Counter(int(p) for p in picks).items()]
    raise ValueError(targeting)


def write_augmented(path, rows):
    """rows: list of dict(id, session_meta, history, current_prompt, label)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def label_of(samples, y, i):
    return y[i]
