"""Failure-mode analysis: where does the current model err? Cross-tab val error
rate by generator (sim/au), session step, and session_meta axes.
"""
import collections
import json
import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data import CLASS_TO_ID, ALL_CLASSES, build_texts, load_samples, split_indices

CKPT = "output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500"
BIAS = "output/pat/ft_BAAI__bge-m3_hist0/logit_bias.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(texts):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding)
    tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, torch_dtype=torch.float16).to(DEV).eval()
    bmap = json.load(open(BIAS))["bias"]
    bias = np.array([bmap.get(c, 0.0) for c in ALL_CLASSES])
    enc = [tok(t, truncation=True, max_length=1024) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    coll = DataCollatorWithPadding(tokenizer=tok)
    out = [0] * len(enc)
    for s in range(0, len(order), 48):
        idx = order[s:s + 48]
        batch = {k: v.to(DEV) for k, v in coll([enc[i] for i in idx]).items()}
        lg = model(**batch).logits.float().cpu().numpy() + bias
        for j, i in enumerate(idx):
            out[i] = int(lg[j].argmax())
    return np.array(out)


def xtab(name, keys, correct):
    """error rate + n per key value, sorted by error rate desc."""
    agg = collections.defaultdict(lambda: [0, 0])   # key -> [wrong, total]
    for k, c in zip(keys, correct):
        agg[k][1] += 1
        agg[k][0] += (0 if c else 1)
    print(f"\n--- error rate by {name} ---")
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0] / kv[1][1])
    for k, (w, t) in rows:
        print(f"  {str(k):22} err={w/t:6.1%}  n={t:6d}")


def main():
    samples, y = load_samples("./data")
    texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
    _, va = split_indices(y, seed=42)
    preds = predict([texts[i] for i in va])
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])
    correct = preds == y_true

    print(f"val n={len(va)}  overall acc={correct.mean():.4f}  "
          f"macro-F1={f1_score(y_true, preds, average='macro'):.4f}")

    S = [samples[i] for i in va]
    gen = [s["id"].split("_")[1] for s in S]
    step = [int(s["id"].split("step_")[1]) for s in S]
    tier = [s["session_meta"]["user_tier"] for s in S]
    lang = [s["session_meta"]["language_pref"] for s in S]
    ci = [s["session_meta"]["workspace"]["last_ci_status"] for s in S]
    dirty = [s["session_meta"]["workspace"]["git_dirty"] for s in S]
    hlen = [min(len(s["history"]) // 3 * 3, 12) for s in S]     # history-length bucket
    stepb = [min(st, 10) for st in step]

    xtab("generator (sim/au)", gen, correct)
    xtab("user_tier", tier, correct)
    xtab("ci_status", ci, correct)
    xtab("git_dirty", dirty, correct)
    xtab("lang_pref", lang, correct)
    xtab("session step (>=10 bucketed)", stepb, correct)
    xtab("history length (bucketed)", hlen, correct)
    xtab("true label", [ALL_CLASSES[t] for t in y_true], correct)

    # per-generator macro-F1 + worst confusions in au
    for g in ("sim", "au"):
        m = np.array([x == g for x in gen])
        print(f"\n[{g}] n={m.sum()} acc={correct[m].mean():.4f} "
              f"macroF1={f1_score(y_true[m], preds[m], average='macro'):.4f}")


if __name__ == "__main__":
    main()
