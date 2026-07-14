"""E46 — aggregate quantization results into the 3 measurement dimensions.

Canonical measurement (per user spec), computed uniformly from saved logits so every
method is scored identically regardless of which run produced it:

  1. change in F1        : macro-F1(quant) and Δ = macro-F1(quant) - macro-F1(fp16 ref)
  2. change in prediction: flip% = fraction of rows whose argmax != fp16 ref argmax
  3. change in logit     : L1 / RMSE / max-abs / cosine of (quant_logits - ref_logits)

Reports all 70k and the 3.5k val slice. Raw logits, no calibration ([[no-logit-calibration]]).
Joins latency/size from _work/results.jsonl. Writes a markdown table to stdout + _work/summary.md.
"""
import glob
import json
import os

import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(__file__)
WORK = os.path.join(HERE, "_work")
REF = os.path.join(WORK, "ref_fp16.npz")

# display order + T4 portability
ORDER = ["fp32", "bf16", "int8_wo", "int8_dyn", "bnb_int8", "bnb_nf4", "bnb_fp4",
         "int4_wo", "fp8_dyn", "trt_fp16", "trt_int8_entropy", "trt_int8_minmax"]
T4 = {"fp32": "yes", "bf16": "yes*", "int8_wo": "yes", "int8_dyn": "yes", "bnb_int8": "yes",
      "bnb_nf4": "yes*", "bnb_fp4": "yes*", "int4_wo": "NO", "fp8_dyn": "NO",
      "trt_fp16": "yes", "trt_int8_entropy": "yes", "trt_int8_minmax": "yes"}
SCHEME = {"fp32": "fp32 (no quant)", "bf16": "bf16 (no quant, dtype baseline)",
          "int8_wo": "torchao int8 W8A16 (bf16)", "int8_dyn": "torchao int8 W8A8 (bf16)",
          "bnb_int8": "bnb LLM.int8() W8A8", "bnb_nf4": "bnb NF4 4-bit wt", "bnb_fp4": "bnb FP4 4-bit wt",
          "int4_wo": "torchao int4 W4A16", "fp8_dyn": "torchao fp8 W8A8 (bf16)",
          "trt_fp16": "TRT fp16 engine", "trt_int8_entropy": "TRT int8 W8A8 entropy-cal",
          "trt_int8_minmax": "TRT int8 W8A8 minmax-cal"}


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def main():
    ref = np.load(REF, allow_pickle=True)
    ref_logits = ref["logits"].astype(np.float32)
    y = ref["y_true"].astype(np.int64)
    vm = ref["val_mask"]           # held-out 3.5k
    tm = ~vm                       # in-sample train (~66.5k, seen during full_data training)
    ref_pred = ref_logits.argmax(-1)
    print(f"split: train(in-sample)={int(tm.sum())}  val(held-out)={int(vm.sum())}")

    # macro-F1 of the fp16 reference, per split (train = inflated in-sample, val = honest)
    f1_ref_tr = f1_score(y[tm], ref_pred[tm], average="macro")
    f1_ref_val = f1_score(y[vm], ref_pred[vm], average="macro")

    # timing/size from the sweep log
    meta = {}
    rj = os.path.join(WORK, "results.jsonl")
    if os.path.exists(rj):
        for line in open(rj):
            r = json.loads(line)
            meta[r["method"]] = r

    rows = []
    # reference row
    rows.append(dict(method="fp16 (REF)", t4="yes", scheme="fp16 shipped", size=622,
                     f1_tr=f1_ref_tr, d_f1_tr=0.0, f1_val=f1_ref_val, d_f1_val=0.0,
                     flip_tr=0.0, flip_val=0.0, l1=0.0, rmse=0.0, maxabs=0.0, cos=1.0,
                     acc_tr=float((ref_pred[tm] == y[tm]).mean()), acc_val=float((ref_pred[vm] == y[vm]).mean()),
                     lat=None))

    found = {os.path.basename(p).replace("logits_", "").replace(".npz", "")
             for p in glob.glob(os.path.join(WORK, "logits_*.npz"))}
    for m in ORDER:
        if m not in found:
            continue
        q = np.load(os.path.join(WORK, f"logits_{m}.npz"))["logits"].astype(np.float32)
        if not np.isfinite(q).all():
            n_nan = int((~np.isfinite(q)).any(-1).sum())
            rows.append(dict(method=m, t4=T4.get(m, "?"), scheme=SCHEME.get(m, m) + " ⚠️NaN",
                             size=meta.get(m, {}).get("size_mb"),
                             f1_tr=float("nan"), d_f1_tr=float("nan"), f1_val=float("nan"),
                             d_f1_val=float("nan"), flip_tr=float("nan"), flip_val=float("nan"),
                             l1=float("nan"), rmse=float("nan"), maxabs=float("nan"), cos=float("nan"),
                             acc_tr=float("nan"), acc_val=float("nan"),
                             lat=f"FAILED: NaN in {n_nan} rows (torchao ref-path, no compiled kernel)"))
            continue
        qp = q.argmax(-1)
        d = q - ref_logits
        pr, pq = softmax(ref_logits), softmax(q)
        num = (ref_logits * q).sum(-1)
        den = np.linalg.norm(ref_logits, axis=-1) * np.linalg.norm(q, axis=-1) + 1e-9
        f1_tr = f1_score(y[tm], qp[tm], average="macro")
        f1_val = f1_score(y[vm], qp[vm], average="macro")
        rows.append(dict(
            method=m, t4=T4.get(m, "?"), scheme=SCHEME.get(m, m), size=meta.get(m, {}).get("size_mb"),
            f1_tr=f1_tr, d_f1_tr=f1_tr - f1_ref_tr, f1_val=f1_val, d_f1_val=f1_val - f1_ref_val,
            flip_tr=100 * (qp[tm] != ref_pred[tm]).mean(), flip_val=100 * (qp[vm] != ref_pred[vm]).mean(),
            l1=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())),
            maxabs=float(np.abs(d).max(-1).mean()), cos=float((num / den).mean()),
            acc_tr=float((qp[tm] == y[tm]).mean()), acc_val=float((qp[vm] == y[vm]).mean()),
            lat=meta.get(m, {}).get("infer_s")))

    # ---- render ----
    hdr = ("| method | scheme | T4 | size MB | **F1 train** | **ΔF1 train** | **F1 val** | **ΔF1 val** "
           "| **flip% train** | **flip% val** | **logit L1** | **logit RMSE** | **max\\|Δ\\|** | **cos** "
           "| acc train | acc val | infer s |")
    sep = "|" + "---|" * 17
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['scheme']} | {r['t4']} | {r['size'] or '—'} "
            f"| {r['f1_tr']:.4f} | {r['d_f1_tr']:+.4f} | {r['f1_val']:.4f} | {r['d_f1_val']:+.4f} "
            f"| {r['flip_tr']:.3f} | {r['flip_val']:.3f} | {r['l1']:.4f} | {r['rmse']:.4f} "
            f"| {r['maxabs']:.3f} | {r['cos']:.5f} | {r['acc_tr']:.4f} | {r['acc_val']:.4f} "
            f"| {r['lat'] or '—'} |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(WORK, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(f"\nref macro-F1: train(in-sample)={f1_ref_tr:.4f} val(held-out)={f1_ref_val:.4f} "
          f"| methods aggregated: {len(rows)-1}")


if __name__ == "__main__":
    main()
