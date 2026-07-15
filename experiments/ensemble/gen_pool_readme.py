"""Generate MODELS.md — the added/excluded reference for the logit pool.
Truth = the .npz files actually present (added) + manifest.jsonl decisions (excluded, deduped).
  PYTHONPATH=. python experiments/ensemble/gen_pool_readme.py
"""
import glob
import json
import os
import re

D = "experiments/ensemble/logits"
OUT = "experiments/ensemble/MODELS.md"   # markdown lives OUTSIDE the data folder

REASON = {
    "skip_leak": "leak — didn't hold out the seed-42 slice (KFold fold / seed 43-44)",
    "skip_incompat": "incompatible arch — loads as garbage (pruned/factored/MLM/smoke)",
    "skip_no_serialize": "no serialize row & not worth guessing (superseded/specialist)",
    "drop_lowf1": "self-F1 < 0.70 (weak / specialist / broken checkpoint)",
    "error": "load error",
}


def expnum(t):
    m = re.search(r"e(\d+)", t)
    if m:
        return f"E{int(m.group(1))}"
    if t.startswith(("t0", "t1", "member", "k6")):
        return "E28/E33"
    if t.startswith("coreset"):
        return "E22"
    if t.startswith("cont_") or t.startswith("bgem3") or t == "supcon" or t == "qwen3_base":
        return "E0/bge"
    if t.startswith("a24") or t.startswith("b24"):
        return "E15"
    if t.startswith("soup"):
        return "E12"
    if t.startswith("granite_names"):
        return "E21"
    return "misc"


def main():
    pool = {os.path.basename(f)[:-4] for f in glob.glob(f"{D}/*.npz")
            if os.path.basename(f) != "_meta.npz"}
    # last decision + self_f1 + serialize per tag from manifest
    info = {}
    for line in open(f"{D}/manifest.jsonl"):
        r = json.loads(line)
        info[r["tag"]] = r   # last write wins
    added, excluded = [], []
    for t, r in info.items():
        if t in pool:
            added.append((expnum(t), t, r.get("self_f1", "?"), r.get("serialize", "?"),
                          r.get("source", "")))
        else:
            added_elsewhere = t in pool
            if not added_elsewhere:
                excluded.append((expnum(t), t, r["decision"]))
    # any pool file with no manifest row (shouldn't happen) -> add
    for t in pool - set(info):
        added.append((expnum(t), t, "?", "?", ""))
    added.sort(key=lambda x: (x[0], -(x[2] if isinstance(x[2], float) else 0)))
    excluded.sort(key=lambda x: (x[2], x[0], x[1]))

    lines = []
    w = lines.append
    w("# Ensemble logit pool — model coverage\n")
    w(f"**{len(added)} models ADDED** (in pool, self-F1 ≥ 0.70) · **{len(excluded)} EXCLUDED**. "
      "Slice = seed-42 3.5k held-out; see [POOL.md](POOL.md). Regenerate: "
      "`python experiments/ensemble/gen_pool_readme.py`.\n")

    # side-by-side two-column summary
    w("## Added vs Excluded (summary)\n")
    w("| ✅ Added (model · self-F1) | ❌ Excluded (model · reason) |")
    w("|---|---|")
    a_disp = [f"`{t}` · {f1}" for _, t, f1, _, _ in added]
    e_disp = [f"`{t}` · {d.replace('skip_','').replace('drop_','')}" for _, t, d in excluded]
    for i in range(max(len(a_disp), len(e_disp))):
        left = a_disp[i] if i < len(a_disp) else ""
        right = e_disp[i] if i < len(e_disp) else ""
        w(f"| {left} | {right} |")

    w("\n## Added — detail\n")
    w("| exp | model | self-F1 | serialize |")
    w("|---|---|---|---|")
    for e, t, f1, ser, _ in added:
        w(f"| {e} | `{t}` | {f1} | {ser} |")

    w("\n## Excluded — detail\n")
    w("| exp | model | decision | why |")
    w("|---|---|---|---|")
    for e, t, d in excluded:
        w(f"| {e} | `{t}` | `{d}` | {REASON.get(d, '')} |")

    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(added)} added, {len(excluded)} excluded")


if __name__ == "__main__":
    main()
