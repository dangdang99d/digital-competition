"""Model soup (E12): average the weights of N independently-trained members that share
the SAME train/val split but differ in init/shuffle (train with `--seed 42 --init_seed k`),
then evaluate the soup on the common eval slice with RAW argmax (NO logit-bias calibration).

Members MUST be the same architecture/recipe (soup averages the full state_dict incl. head).
The eval slice is reconstructed exactly like finetune.py: split_indices(seed) then, for
--full_data, the 25% nested holdout (random_state=seed) — the untouched slice all members
were evaluated on. Reports each member's own macro-F1, the soup's, and the delta vs best-single.

Usage:
  python -m analysis.soup_average \
    --members <ckptA> <ckptB> <ckptC> \
    --serialize richargs --full_data --max_len 512 --seed 42 \
    --out_dir output/pat/soup_granite_ls
"""
import argparse
import os

import numpy as np
import torch

from src.data import (CLASS_TO_ID, build_texts, load_samples, macro_f1,
                      split_indices)
from src.runlog import log_cmd


def eval_slice_indices(y, seed, full_data):
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=seed)
    if full_data:
        from sklearn.model_selection import train_test_split
        _, va = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=seed)
    return va


def predict(model_dir, texts, device):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.truncation_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True).to(device).eval()
    enc = [tok(t, truncation=True, max_length=MAX_LEN) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    coll = DataCollatorWithPadding(tokenizer=tok)
    out = [None] * len(enc)
    with torch.no_grad():
        for s in range(0, len(order), 64):
            idx = order[s:s + 64]
            batch = {k: v.to(device) for k, v in coll([enc[i] for i in idx]).items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                lg = model(**batch).logits.float().argmax(-1).cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = int(lg[j])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.array(out)


def average_state_dicts(members, out_dir):
    """Uniform mean of the members' float params; saved to out_dir (member[0] = template)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # load via from_pretrained to get proper state dicts (safetensors handled internally)
    sds = []
    for m in members:
        sd = AutoModelForSequenceClassification.from_pretrained(
            m, local_files_only=True).state_dict()
        sds.append({k: v.float() for k, v in sd.items()})
    avg = {k: sum(sd[k] for sd in sds) / len(sds) for k in sds[0]}
    model = AutoModelForSequenceClassification.from_pretrained(members[0], local_files_only=True)
    model.load_state_dict({k: avg[k].to(model.state_dict()[k].dtype) for k in avg})
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(members[0], local_files_only=True).save_pretrained(out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="member checkpoint dirs")
    ap.add_argument("--serialize", default="richargs")
    ap.add_argument("--full_data", action="store_true")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42, help="the SPLIT seed (shared by all members)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    log_cmd()

    global MAX_LEN
    MAX_LEN = args.max_len
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples, y = load_samples(args.data_dir)
    va = eval_slice_indices(y, args.seed, args.full_data)
    texts = build_texts([samples[i] for i in va], input_mode="context",
                        max_hist=None, variant=args.serialize)
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])
    print(f"eval slice n={len(va)}  members={len(args.members)}  serialize={args.serialize}")

    per_member = []
    for m in args.members:
        f1 = macro_f1(y_true, predict(m, texts, device))
        per_member.append(f1)
        print(f"  member {os.path.basename(os.path.dirname(m)) or m}: {f1:.5f}")

    soup_dir = average_state_dicts(args.members, args.out_dir)
    soup_f1 = macro_f1(y_true, predict(soup_dir, texts, device))

    best = max(per_member)
    print("\n==== SOUP RESULT ====")
    print(f"members: {[round(f, 5) for f in per_member]}  (best single = {best:.5f})")
    print(f"SOUP:    {soup_f1:.5f}   Δ vs best-single = {soup_f1 - best:+.5f}")
    print(f"saved soup -> {soup_dir}")


if __name__ == "__main__":
    main()
