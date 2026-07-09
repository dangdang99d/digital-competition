"""Palu-style KV-SVD degradation probe on the trained Qwen3-0.6B classifier.

Training-free: replace each k_proj/v_proj weight with its WHITENED (activation-
aware, SVD-LLM style) truncated-SVD reconstruction and measure val macro-F1 drop
across rank-keep ratios. No fine-tune — this is the pure low-rank compression cost
before any recovery. Whitening: minimize ||(W-Ŵ)X||, i.e. truncate (W·L) then
un-whiten, where L is the Cholesky factor of the input second-moment S=E[xxᵀ]
(collected from a calibration pass). Plain weight-SVD would minimize ||W-Ŵ||.

Usage: python -m analysis.palu_probe [--max_len 512] [--ratios 0.875,0.75,0.5,0.375,0.25]
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data import CLASS_TO_ID, build_texts, load_samples, split_indices

CKPT = "output/pat/ft_Qwen__Qwen3-Embedding-0.6B/checkpoint-10500"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_tok():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, torch_dtype=torch.float16).to(DEV).eval()
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


@torch.no_grad()
def predict(model, tok, enc, bs=24):
    from transformers import DataCollatorWithPadding
    coll = DataCollatorWithPadding(tokenizer=tok)
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]), reverse=True)
    out = [0] * len(enc)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = {k: v.to(DEV) for k, v in coll([enc[i] for i in idx]).items()}
        pr = model(**batch).logits.float().argmax(-1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = int(pr[j])
    return np.array(out)


@torch.no_grad()
def collect_grams(model, tok, calib_enc, bs=8):
    """Input second-moment S=Σ xxᵀ for each k_proj/v_proj (its actual input tensor)."""
    from transformers import DataCollatorWithPadding
    coll = DataCollatorWithPadding(tokenizer=tok)
    targets = {n: m for n, m in model.named_modules() if n.endswith(("k_proj", "v_proj"))}
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
        model(**batch)
    for h in hooks:
        h.remove()
    for n in grams:
        grams[n] /= max(1, cnt[n])
    return grams, targets


def whitened_trunc(W, S, r):
    """Ŵ = truncate_r(W·L)·L⁻¹, S = L·Lᵀ. Returns (Ŵ fp32, energy_kept)."""
    W = W.double()
    dm = S.diag().mean()
    for ridge in (1e-4, 1e-3, 1e-2, 1e-1):
        try:
            L = torch.linalg.cholesky(S + ridge * dm * torch.eye(S.shape[0], device=S.device, dtype=S.dtype))
            break
        except Exception:
            continue
    M = W @ L
    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    energy = (s[:r].pow(2).sum() / s.pow(2).sum()).item()
    Linv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    What = (U[:, :r] * s[:r]) @ Vh[:r] @ Linv
    return What, energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--ratios", default="0.875,0.75,0.5,0.375,0.25")
    ap.add_argument("--n_val", type=int, default=0, help="0 = full val")
    ap.add_argument("--n_calib", type=int, default=256)
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    samples, y = load_samples("./data")
    texts = build_texts(samples, input_mode="context", max_hist=None, variant="v1")
    tr, va = split_indices(y, seed=42)
    if args.n_val:
        va = va[:args.n_val]
    y_true = np.array([CLASS_TO_ID[y[i]] for i in va])

    model, tok = load_model_tok()
    val_enc = [tok(texts[i], truncation=True, max_length=args.max_len) for i in va]
    calib_enc = [tok(texts[i], truncation=True, max_length=args.max_len) for i in tr[:args.n_calib]]

    base_f1 = f1_score(y_true, predict(model, tok, val_enc), average="macro")
    print(f"baseline (rank 1.0)   macro-F1 = {base_f1:.4f}   [expect ~0.768]")

    grams, targets = collect_grams(model, tok, calib_enc)
    orig = {n: m.weight.data.clone() for n, m in targets.items()}
    full_rank = next(iter(targets.values())).weight.shape[1]

    print(f"\n{'ratio':>6} {'rank':>5} {'energy':>7} {'macro-F1':>9} {'Δ':>8}")
    for ratio in [float(x) for x in args.ratios.split(",")]:
        r = max(1, int(round(ratio * full_rank)))
        energies = []
        for n, m in targets.items():
            What, e = whitened_trunc(orig[n].float(), grams[n], r)
            m.weight.data = What.to(m.weight.dtype)
            energies.append(e)
        f1 = f1_score(y_true, predict(model, tok, val_enc), average="macro")
        print(f"{ratio:>6.3f} {r:>5d} {np.mean(energies):>6.1%} {f1:>9.4f} {f1 - base_f1:>+8.4f}")
        for n, m in targets.items():           # restore for next ratio
            m.weight.data = orig[n].clone()


if __name__ == "__main__":
    main()
