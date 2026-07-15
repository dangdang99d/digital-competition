"""FFN (and KV) whitened-SVD degradation probe on the champion Qwen3-0.6B classifier.

Training-free extension of analysis/palu_probe.py to the MLP projections
(gate_proj / up_proj / down_proj), the real compression prize (~44% of params).
Same WHITENED (activation-aware, SVD-LLM style) truncated-SVD: replace each
projection weight W with truncate_r(W.L).L^-1 where S=L.L^T is the input
second-moment collected from a calibration pass. Sweep the rank-keep ratio and
measure uncalibrated val macro-F1 drop. No fine-tune -- pure low-rank cost.

Runs on the VOCAB-PRUNED champion (submit_0703_qwen3_pruned): the transformer
layers + head are byte-identical to the full champion; only input embeddings are
sliced, so remap.npy is applied to input_ids to reproduce identical hidden
states (and thus identical FFN activations / logits) as the full model.

Usage:
  python -m analysis.palu_probe_ffn --model_dir <unzipped>/model/qwen3-0.6b \
      --proj both --n_val 3000 --ratios 1.0,0.875,0.75,0.625,0.5,0.375,0.25
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data import CLASS_TO_ID, build_texts, load_samples, split_indices

DEV = "cuda" if torch.cuda.is_available() else "cpu"

PROJ_SETS = {
    "kv":  ("k_proj", "v_proj"),
    "ffn": ("gate_proj", "up_proj", "down_proj"),   # qwen3 SwiGLU
    "ffn_mb": ("Wi", "Wo"),                          # granite/ModernBERT GeGLU
}


def load_model_tok(model_dir):
    # Mirror the verified submission script.py EXACTLY: pad_token=eos-if-None and
    # DO NOT touch model.config.pad_token_id. The pruned config ships
    # pad_token_id=25284 (a PRUNED-space row = remap[eos_full]); after remap the
    # padded positions hold 25284, so Qwen3's last-token pooling finds the right
    # end token. Overriding config.pad_token_id to the full-space eos (151643)
    # breaks pooling -> garbage logits on padded rows (~0.10 macro-F1 loss).
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:  # granite/ModernBERT: eager attn (no compiled embeddings / flash for hooks)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float16,
            attn_implementation="eager", reference_compile=False).to(DEV).eval()
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float16).to(DEV).eval()
    remap = None
    if os.path.exists(f"{model_dir}/remap.npy"):   # vocab-pruned (qwen3); granite = full vocab
        remap = torch.from_numpy(np.load(f"{model_dir}/remap.npy")).long().to(DEV)
        pad_full = tok.pad_token_id
        print(f"config.pad_token_id={model.config.pad_token_id}  tok.pad(full)={pad_full}  "
              f"remap[pad_full]={int(remap[pad_full])}  (pooling OK iff ==config.pad_token_id)")
    return model, tok, remap


@torch.no_grad()
def predict(model, tok, enc, remap, bs=24):
    from transformers import DataCollatorWithPadding
    coll = DataCollatorWithPadding(tokenizer=tok)
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    out = [0] * len(enc)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = {k: v.to(DEV) for k, v in coll([enc[i] for i in idx]).items()}
        if remap is not None:
            batch["input_ids"] = remap[batch["input_ids"]]  # pruned-vocab remap
        pr = model(**batch).logits.float().argmax(-1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = int(pr[j])
    return np.array(out)


@torch.no_grad()
def collect_grams(model, tok, calib_enc, targets, remap, bs=8):
    """Input second-moment S=E[x x^T] for each target module (its actual input)."""
    from transformers import DataCollatorWithPadding
    coll = DataCollatorWithPadding(tokenizer=tok)
    grams = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1], dtype=torch.float64, device=DEV)
             for n, m in targets.items()}
    cnt = {n: 0 for n in targets}

    def mk(n):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            grams[n] += x.t() @ x
            cnt[n] += x.shape[0]
        return hook

    hooks = [m.register_forward_hook(mk(n)) for n, m in targets.items()]
    for s in range(0, len(calib_enc), bs):
        batch = {k: v.to(DEV) for k, v in coll(calib_enc[s:s + bs]).items()}
        if remap is not None:
            batch["input_ids"] = remap[batch["input_ids"]]
        model(**batch)
    for h in hooks:
        h.remove()
    for n in grams:
        grams[n] /= max(1, cnt[n])
    return grams


def whitened_svd(W, S):
    """Precompute whitened-SVD components once: S=L.L^T, M=W.L=U.diag(s).Vh, Linv=L^-1.
    Truncating to any rank r is then a cheap slice (see reconstruct)."""
    W = W.double()
    dm = S.diag().mean()
    L = None
    for ridge in (1e-4, 1e-3, 1e-2, 1e-1):
        try:
            L = torch.linalg.cholesky(S + ridge * dm * torch.eye(S.shape[0], device=S.device, dtype=S.dtype))
            break
        except Exception:
            continue
    M = W @ L
    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    Linv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    return U, s, Vh, Linv


def reconstruct(comp, r):
    """Ŵ = truncate_r(W.L).L^-1 from cached components. Returns (What fp32, energy)."""
    U, s, Vh, Linv = comp
    r = min(r, s.shape[0])
    energy = (s[:r].pow(2).sum() / s.pow(2).sum()).item()
    What = (U[:, :r] * s[:r]) @ Vh[:r] @ Linv
    return What.float(), energy


def param_frac(out_dim, in_dim, r):
    """Fraction of the matrix's params kept if factorized to rank r (two matmuls)."""
    return r * (out_dim + in_dim) / (out_dim * in_dim)


def run_sweep(model, tok, val_enc, y_true, base_f1, grams, targets, orig, ratios, remap, label):
    rows = []
    comps = {n: whitened_svd(orig[n].float(), grams[n]) for n in targets}   # SVD once per module
    print(f"\n[{label}]  {'ratio':>6} {'rank':>5} {'energy':>7} {'param%':>7} {'macro-F1':>9} {'delta':>8}")
    for ratio in ratios:
        energies, pfracs, rs = [], [], []
        for n, m in targets.items():
            out_dim, in_dim = m.weight.shape
            mn = min(out_dim, in_dim)
            r = min(max(1, int(round(ratio * mn))), mn)
            What, e = reconstruct(comps[n], r)
            m.weight.data = What.to(m.weight.dtype)
            energies.append(e)
            pfracs.append(min(1.0, param_frac(out_dim, in_dim, r)))
            rs.append(r)
        f1 = f1_score(y_true, predict(model, tok, val_enc, remap), average="macro")
        row = dict(ratio=ratio, rank=int(np.mean(rs)), energy=float(np.mean(energies)),
                   param=float(np.mean(pfracs)), f1=float(f1), delta=float(f1 - base_f1))
        rows.append(row)
        print(f"[{label}]  {ratio:>6.3f} {row['rank']:>5d} {row['energy']:>6.1%} "
              f"{row['param']:>6.1%} {f1:>9.4f} {f1 - base_f1:>+8.4f}")
        for n, m in targets.items():           # restore for next ratio
            m.weight.data = orig[n].clone()
    del comps
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--proj", default="both", choices=["kv", "ffn", "both"])
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--ratios", default="1.0,0.875,0.75,0.625,0.5,0.375,0.25")
    ap.add_argument("--n_val", type=int, default=3000)
    ap.add_argument("--n_calib", type=int, default=256)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()
    ratios = [float(x) for x in args.ratios.split(",")]

    samples, y = load_samples("./data")
    texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
    tr, va = split_indices(y, seed=42)
    if args.n_val:
        va = va[:args.n_val]
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])

    model, tok, remap = load_model_tok(args.model_dir)
    val_enc = [tok(texts[i], truncation=True, max_length=args.max_len) for i in va]
    calib_enc = [tok(texts[i], truncation=True, max_length=args.max_len) for i in tr[:args.n_calib]]

    base_f1 = f1_score(y_true, predict(model, tok, val_enc, remap), average="macro")
    print(f"baseline (unmodified) macro-F1 = {base_f1:.4f}   [KV probe subset ref ~0.7734]")

    # auto-detect backbone: granite/ModernBERT has mlp.Wi/Wo (GeGLU) and fused Wqkv attn
    is_modernbert = any(n.endswith(("mlp.Wi", "mlp.Wo")) for n, _ in model.named_modules())
    if args.proj == "both":
        proj_sets = ["ffn_mb"] if is_modernbert else ["kv", "ffn"]
    elif args.proj == "ffn" and is_modernbert:
        proj_sets = ["ffn_mb"]
    else:
        proj_sets = [args.proj]
    suffixes = tuple(sum((PROJ_SETS[p] for p in proj_sets), ()))
    targets_all = {n: m for n, m in model.named_modules() if n.endswith(suffixes)}
    grams = collect_grams(model, tok, calib_enc, targets_all, remap)
    orig = {n: m.weight.data.clone() for n, m in targets_all.items()}

    results = {"baseline": base_f1, "curves": {}}
    for p in proj_sets:
        tgt = {n: m for n, m in targets_all.items() if n.endswith(PROJ_SETS[p])}
        results["curves"][p] = run_sweep(model, tok, val_enc, y_true, base_f1,
                                         grams, tgt, orig, ratios, remap, p)

    if args.out_json:
        import json
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
