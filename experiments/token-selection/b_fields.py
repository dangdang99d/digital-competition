"""E24 method B — ONE source of truth for field transforms, shared by Phase-0 (occlusion)
and Phase-1 (reduced-input build) so the two never drift.

A "field" is one droppable piece of the richargs serialization. Two kinds:
  - KWARG_OCC : produced by serialize() kwargs (whole meta block · all history · history depth ·
                action detail) — exact, no string parsing.
  - MASK_OCC  : produced by removing a token/line from the serialized text (individual meta
                subfields · user turns · prompt) — meta values are space-free so token-drop is clean.
"""
from src.data import serialize

RICHARGS = dict(rich_meta=True, arg_basenames=True)                      # champion serialization
META_FIELDS = ["tier", "lang", "turn", "budget", "elapsed", "codelang",
               "loc", "ci", "dirty", "open", "files"]


def mask_meta_subfield(text, field):
    """Drop the `field=<value>` token from the first (meta) line only (values are space-free)."""
    lines = text.split("\n")
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        toks = [t for t in lines[0][1:-1].split(" ") if not t.startswith(field + "=")]
        lines[0] = "[" + " ".join(toks) + "]"
    return "\n".join(lines)


def drop_lines(text, prefix):
    return "\n".join(l for l in text.split("\n") if not l.startswith(prefix))


KWARG_OCC = {
    "meta_block":    dict(rich_meta=False, drop_meta=True, arg_basenames=True),
    "history_all":   dict(**RICHARGS, strip_history=True),
    "hist_depth<=8": dict(**RICHARGS, max_hist=8),
    "hist_depth<=4": dict(**RICHARGS, max_hist=4),
    "action_detail": dict(**RICHARGS, lean_actions=True),
    "action_bare":   dict(**RICHARGS, lean_actions=True, bare_actions=True),
}
MASK_OCC = {f"meta.{f}": (lambda fld: (lambda t: mask_meta_subfield(t, fld)))(f) for f in META_FIELDS}
MASK_OCC["user_turns"] = lambda t: drop_lines(t, "USER:")
MASK_OCC["prompt"] = lambda t: drop_lines(t, "PROMPT:")

ALL_OCCLUSIONS = list(KWARG_OCC) + list(MASK_OCC)


def serialize_occluded(sample, name):
    """richargs text for `sample` with a SINGLE occlusion applied (Phase-0)."""
    if name in KWARG_OCC:
        return serialize(sample, **KWARG_OCC[name])
    return MASK_OCC[name](serialize(sample, **RICHARGS))


def apply_drops(sample, drop_names):
    """richargs text with MULTIPLE fields dropped (Phase-1 kept-set): kwarg-drops merged into one
    serialize() call, then mask-drops applied to the result in order."""
    kw = dict(RICHARGS)
    masks = []
    for n in drop_names:
        if n in KWARG_OCC:
            kw.update(KWARG_OCC[n])
        elif n in MASK_OCC:
            masks.append(MASK_OCC[n])
        else:
            raise KeyError(f"unknown field '{n}'; valid: {ALL_OCCLUSIONS}")
    text = serialize(sample, **kw)
    for m in masks:
        text = m(text)
    return text
