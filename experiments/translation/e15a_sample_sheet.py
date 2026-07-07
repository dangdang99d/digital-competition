"""Build a human-eyeball sample sheet (~20 orig->trans pairs) from e15a_pairs.json.

Selects a spread: pairs that carry code spans (the risky ones) prioritized, plus a
few plain-prose pairs, so the reviewer sees both quality and code-preservation.
Writes experiments/translation/e15a_sample_sheet.md.
"""
import json, os, re

OUT = "experiments/translation"
SPAN_PATTERNS = [
    re.compile(r"(?:[A-Za-z0-9_]+/)+[A-Za-z0-9_.\*]+"),
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_.]*"),
    re.compile(r"(?<![A-Za-z])[a-z]+(?:[A-Z][a-z0-9]*)+"),
    re.compile(r"(?<![A-Za-z])[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"),
    re.compile(r"(?<![A-Za-z0-9_])[a-z0-9]+_[a-z0-9_]+"),
    re.compile(r"(?<![A-Za-z0-9_])[A-Z]{2,}(?:_[A-Z0-9]+)*(?![A-Za-z0-9_])"),
    re.compile(r"`[^`]+`"),
    re.compile(r"(?<![A-Za-z0-9])[1-5]\d{2}(?![A-Za-z0-9])"),
]


def spans(text):
    raw = set()
    for pat in SPAN_PATTERNS:
        for m in pat.finditer(text):
            s = m.group(0).strip("`")
            if len(s) >= 2:
                raw.add(s)
    return {s for s in raw if not any(s != o and s in o for o in raw)}


def main():
    pairs = json.load(open(os.path.join(OUT, "e15a_pairs.json"), encoding="utf-8"))
    coded, plain = [], []
    for p in pairs:
        sp = spans(p["orig"])
        (coded if sp else plain).append((p, sp))
    # 15 code-bearing (sorted by span count desc for density) + 5 plain
    coded.sort(key=lambda x: -len(x[1]))
    picked = coded[:15] + plain[:5]

    lines = ["# E15a sample sheet — NLLB-200-distilled-600M  KO->EN",
             "",
             "20 real (original -> translation) pairs. `CODE SPANS` lists the "
             "must-be-verbatim tokens found in the source; `KEPT?` marks whether "
             "all of them survived the translation verbatim.",
             ""]
    for i, (p, sp) in enumerate(picked, 1):
        kept = all(s in p["trans"] for s in sp) if sp else True
        mark = "YES" if kept else "**NO**"
        missing = [s for s in sp if s not in p["trans"]]
        lines += [
            f"### {i}. [{p['field']}]  code-spans-kept: {mark}",
            f"- **KO  :** {p['orig']}",
            f"- **EN  :** {p['trans']}",
        ]
        if sp:
            lines.append(f"- **spans:** `{'`, `'.join(sorted(sp))}`"
                         + (f"  — MISSING: `{'`, `'.join(missing)}`" if missing else ""))
        lines.append("")
    open(os.path.join(OUT, "e15a_sample_sheet.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", os.path.join(OUT, "e15a_sample_sheet.md"), "with", len(picked), "pairs")


if __name__ == "__main__":
    main()
