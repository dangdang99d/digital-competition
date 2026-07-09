"""What distinguishes the classes WITHIN each confusion group?

For each group (explore / execute / non-code), contrast each class
against its siblings on the 70k train set:
  1. current_prompt tokens with the highest log-odds for the class (vs rest of
     group), min frequency 30 — the phrasing conventions.
  2. metadata skews: ci status, git_dirty, open_files, lang, turn bucket.
  3. last history action distribution — transition conventions.

Usage: python -m analysis.group_discriminators [--group explore|execute|non-code|all]
"""
import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

GROUPS = {
    "explore": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "non-code": ["ask_user", "plan_task", "web_search"],
}

_TOKEN_RE = re.compile(r"[A-Za-z_./*]+|[가-힣]+")


def tokens_of(text):
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def top_log_odds(cls_counts, rest_counts, cls_total, rest_total, min_count=30, k=15):
    out = []
    for tok, c in cls_counts.items():
        r = rest_counts.get(tok, 0)
        if c + r < min_count:
            continue
        lo = np.log((c + 0.5) / (cls_total + 0.5)) - np.log((r + 0.5) / (rest_total + 0.5))
        out.append((lo, tok, c, r))
    out.sort(reverse=True)
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=list(GROUPS) + ["all"])
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--top_k", type=int, default=15)
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    import csv
    samples = [json.loads(l) for l in open(f"{args.data_dir}/train.jsonl", encoding="utf-8")]
    labels = {r["id"]: r["action"] for r in csv.DictReader(open(f"{args.data_dir}/train_labels.csv", encoding="utf-8"))}

    for gname, classes in GROUPS.items():
        if args.group not in ("all", gname):
            continue
        subset = [(s, labels[s["id"]]) for s in samples if labels[s["id"]] in classes]
        print(f"\n{'='*80}\nGROUP: {gname}  (n={len(subset)})\n{'='*80}")

        tok_counts = {c: Counter() for c in classes}
        tok_totals = {c: 0 for c in classes}
        meta = {c: defaultdict(Counter) for c in classes}
        last_act = {c: Counter() for c in classes}
        n_cls = Counter()

        for s, lbl in subset:
            n_cls[lbl] += 1
            ts = tokens_of(s["current_prompt"])
            tok_counts[lbl].update(set(ts))          # presence, not multiplicity
            tok_totals[lbl] += 1
            sm, ws = s["session_meta"], s["session_meta"]["workspace"]
            meta[lbl]["ci"][ws["last_ci_status"]] += 1
            meta[lbl]["dirty"][ws["git_dirty"]] += 1
            meta[lbl]["open"][len(ws["open_files"]) > 0] += 1
            meta[lbl]["lang"][sm["language_pref"]] += 1
            meta[lbl]["turn"]["first" if sm["turn_index"] <= 1 else "later"] += 1
            la = next((t["name"] for t in reversed(s["history"]) if t.get("role") != "user"), "<none>")
            last_act[lbl][la] += 1

        for c in classes:
            rest = [x for x in classes if x != c]
            rest_counts = Counter()
            for r in rest:
                rest_counts.update(tok_counts[r])
            rest_total = sum(tok_totals[r] for r in rest)
            print(f"\n--- {c}  (n={n_cls[c]}) ---")
            tops = top_log_odds(tok_counts[c], rest_counts, tok_totals[c], rest_total, k=args.top_k)
            print("  distinctive prompt tokens (log-odds vs siblings):")
            for lo, tok, cc, rc in tops:
                print(f"    {lo:+.2f}  {tok!r:30s} (in {cc}/{n_cls[c]} of {c}, {rc}/{rest_total} of siblings)")
            print("  metadata skew (class % vs sibling %):")
            for feat in ("ci", "dirty", "open", "turn"):
                mine = meta[c][feat]
                tot = sum(mine.values())
                sibs = Counter()
                for r in rest:
                    sibs.update(meta[r][feat])
                stot = sum(sibs.values())
                diffs = []
                for kk in set(mine) | set(sibs):
                    pm, ps = mine.get(kk, 0) / tot, sibs.get(kk, 0) / stot
                    if abs(pm - ps) >= 0.03:
                        diffs.append(f"{feat}={kk}: {pm:.0%} vs {ps:.0%}")
                if diffs:
                    print(f"    {'; '.join(diffs)}")
            mine = last_act[c]
            tot = sum(mine.values())
            sibs = Counter()
            for r in rest:
                sibs.update(last_act[r])
            stot = sum(sibs.values())
            diffs = []
            for kk in set(mine) | set(sibs):
                pm, ps = mine.get(kk, 0) / tot, sibs.get(kk, 0) / stot
                if abs(pm - ps) >= 0.02:
                    diffs.append((pm - ps, f"last_action={kk}: {pm:.0%} vs {ps:.0%}"))
            if diffs:
                print("  last-history-action skew:")
                for _, d in sorted(diffs, reverse=True)[:6]:
                    print(f"    {d}")


if __name__ == "__main__":
    main()
