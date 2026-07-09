"""E24 method B — Phase 0 (training-free): rank serialized FIELDS by occlusion importance.

For each field, remove it from the richargs serialization, evaluate the FROZEN champion's
macro-F1 on the SAME 3.5k held-out slice, report ΔF1 = baseline - occluded.
  ΔF1 big  -> important field (KEEP)
  ΔF1 ~0   -> droppable noise (candidate to DROP in Phase 1)
  ΔF1 < 0  -> field HURTS (distractor -> definitely drop)

fp32 so BASELINE ≈ champion 0.7803 (also validates the held-out-slice replication).
Field transforms come from b_fields.py (shared with Phase-1). No shared code touched.
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python experiments/token-selection/b_field_occlusion.py
"""
import argparse, os, json, numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from src.data import load_samples, split_indices, serialize, macro_f1, CLASS_TO_ID, ALL_CLASSES
from b_fields import RICHARGS, ALL_OCCLUSIONS, serialize_occluded

CKPT = ("output/pat/ft_ibm-granite__granite-embedding-311m-multilingual-r2"
        "_e8a_ls_richargs_full/checkpoint-8314")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--data", default="data")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke: use only first N held-out")
    ap.add_argument("--out", default="experiments/token-selection/artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.ckpt, num_labels=len(ALL_CLASSES), torch_dtype=torch.float32,
        trust_remote_code=True).to(dev).eval()

    samples, labels = load_samples(args.data)
    y_ids = np.array([CLASS_TO_ID[a] for a in labels])
    tr, va = split_indices(labels, seed=42)                              # labels=strings (match finetune)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=y_ids[va], random_state=42)
    if args.limit:
        va_eval = va_eval[:args.limit]
    samp = [samples[i] for i in va_eval]
    ytrue = y_ids[va_eval]
    print(f"held-out eval slice: n={len(va_eval)}")

    @torch.no_grad()
    def eval_f1(texts):
        preds = []
        for b in range(0, len(texts), args.batch_size):
            enc = tok(texts[b:b + args.batch_size], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
        return macro_f1(ytrue, np.concatenate(preds))

    f1_base = eval_f1([serialize(r, **RICHARGS) for r in samp])
    print(f"\nBASELINE richargs macro-F1 = {f1_base:.4f}  (champion ref 0.7803 on full 3.5k)")

    rows = []
    for name in ALL_OCCLUSIONS:
        f1 = eval_f1([serialize_occluded(r, name) for r in samp])
        rows.append((name, f1, f1_base - f1))
    rows.sort(key=lambda r: r[2], reverse=True)                          # most-important first

    print(f"\n{'field':>16} {'kept_F1':>9} {'dF1':>9}   (dF1>0 HELPS; ~0 droppable; <0 HURTS)")
    for name, f1, d in rows:
        print(f"{name:>16} {f1:9.4f} {d:+9.4f}")
    out = os.path.join(args.out, "b_field_importance.json")
    json.dump({"baseline": float(f1_base),
               "fields": [{"field": n, "kept_f1": float(f1), "dF1": float(d)} for n, f1, d in rows]},
              open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
