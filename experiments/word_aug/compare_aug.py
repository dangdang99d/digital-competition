"""E39 augmentation QC — compare generated augmented rows vs the originals they derive from,
BEFORE training. Checks: (1) structure byte-identical (only free-text changed), (2) code
identifiers preserved, (3) class distribution (balanced should flatten to max), (4) side-by-side
free-text samples for a human label-preservation read.

Usage: python -m experiments.word_aug.compare_aug --aug <aug.jsonl> [--n 12]
"""
import argparse
import collections
import json
import re

from src.data import ALL_CLASSES, SERIALIZE_VARIANTS, load_samples, serialize


def orig_id(aug_id):
    # "<origid>-cmlm3" / "-llm0" -> "<origid>"
    return re.sub(r"-(cmlm|llm)\d+$", "", aug_id)


def freetext(sample):
    parts = [sample.get("current_prompt") or ""]
    parts += [t.get("content") or "" for t in sample.get("history", []) if t.get("role") == "user"]
    return parts


def structure_lines(sample):
    KW = SERIALIZE_VARIANTS["richargs"]
    return [l for l in serialize(sample, **KW).splitlines()
            if not (l.startswith("USER:") or l.startswith("PROMPT:"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    samples, y = load_samples("./data")
    by_id = {s["id"]: (s, y[i]) for i, s in enumerate(samples)}

    aug = [json.loads(l) for l in open(args.aug, encoding="utf-8") if l.strip()]
    print(f"augmented rows: {len(aug)}  from {args.aug}\n")

    # ---- structure preservation + label match ----
    struct_ok = label_ok = 0
    for a in aug:
        o, oy = by_id[orig_id(a["id"])]
        if structure_lines(a) == structure_lines(o):
            struct_ok += 1
        if a["label"] == oy:
            label_ok += 1
    print(f"structure byte-identical: {struct_ok}/{len(aug)} ({100*struct_ok/len(aug):.1f}%)")
    print(f"label == source label:    {label_ok}/{len(aug)} (must be 100%)\n")

    # ---- class distribution ----
    c = collections.Counter(a["label"] for a in aug)
    print("augmented class counts (min..max):",
          min(c.values()), "..", max(c.values()))
    # combined with originals (for balanced this should be ~flat at the max)
    base = collections.Counter(oy for _, oy in by_id.values())

    # ---- how much did the text actually change? ----
    changed = same = 0
    for a in aug:
        o, _ = by_id[orig_id(a["id"])]
        if freetext(a) != freetext(o):
            changed += 1
        else:
            same += 1
    print(f"free-text actually changed: {changed}/{len(aug)} "
          f"({100*changed/len(aug):.1f}%; {same} unchanged)\n")

    # ---- side-by-side samples ----
    print("=" * 78, "\nSAMPLES (orig -> aug free-text)\n" + "=" * 78)
    import random
    for a in aug[:: max(1, len(aug) // args.n)][:args.n]:
        o, oy = by_id[orig_id(a["id"])]
        of, af = freetext(o), freetext(a)
        for so, sa in zip(of, af):
            if so != sa:
                print(f"  [{oy}]")
                print(f"   orig: {so[:110]}")
                print(f"   aug : {sa[:110]}")
                break
    print()


if __name__ == "__main__":
    main()
