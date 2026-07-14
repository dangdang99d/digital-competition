"""E46 — inference-script upgrade screen on the shared 3.5k held-out slice.

Model = e38_t031fd (the weights inside submit_0714_awp_t031.zip, pre vocab-prune;
parity-gated at zip build, so identical predictions). Slice + serialization match
experiments/ensemble/harvest_new.py exactly (richargs, max_len 512, bs32, dataset
order) and the baseline is asserted against the cached pool logits e38_t031fd.npz.

Arms (all label-free at "test time"; hyperparam grids with label-based picks use
the honest 2-fold protocol):
  base      fp32 forward (parity check vs pool npz)
  fp16      full fp16 body+head (ship precision) / fp16 body + fp32 head+classifier
  mcdrop    OFF-LABEL probe: model trained with ALL dropout p=0.0 -> inject p at
            inference (nn.Dropout modules only), K passes, mean softmax
  shot      E23 SHOT/IM reference: entropy-min + IM diversity on encoder LN affines
  sar       SAR-style: entropy filter (H < 0.4*ln C) + SAM (rho=0.05) on LN affines
  t3a       T3A prototype classifier on penultimate features (backprop-free), online
  knn       kNN label propagation over slice embeddings; grid picked honest-2-fold

Caveat printed with results: the slice is CLEAN (no train->test shift), so the
adaptation arms (shot/sar/t3a) are expected ~0/negative here (E23 precedent:
-0.0006..-0.0034 local -> +0.0019 LB). Their slice job is implementation + collapse
safety + clean-data cost; LB decides gain. mcdrop/knn/fp16 are honestly measurable.
"""
import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, macro_f1, split_indices
from src.runlog import log_cmd

CKPT = ("output/e38/promote_top4/"
        "ft_ibm-granite__granite-embedding-311m-multilingual-r2_e38_t031fd")
POOL_NPZ = "experiments/ensemble/logits/e38_t031fd.npz"
OUT_CSV = "experiments/performance-boost/e46_infer_upgrades.csv"
OUT_NPZ = "experiments/performance-boost/e46_infer_upgrades_preds.npz"
C = 14


# ---------------------------------------------------------------- data / model
def load_slice(data_dir="./data"):
    samples, labels = load_samples(data_dir)
    yid = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = split_indices(labels, seed=42)
    _, va_eval = train_test_split(va, test_size=0.25, stratify=yid[va], random_state=42)
    meta = np.load("experiments/ensemble/logits/_meta.npz")
    assert np.array_equal(meta["va_idx"], va_eval), "slice drift vs _meta.npz!"
    texts = build_texts([samples[i] for i in va_eval], variant="richargs")
    return texts, yid[va_eval]


def fresh_model(device, dtype=torch.float32):
    # reference_compile=False: ModernBERT's compiled embedding path caches the dtype
    # of the FIRST forward; a later .half()/precision change then feeds fp32 hidden
    # states into fp16 weights (the E46 v1 crash). Eager embeddings avoid the trap.
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, torch_dtype=dtype, trust_remote_code=True,
        reference_compile=False).to(device).eval()
    return model


@torch.no_grad()
def forward_all(model, tok, texts, device, bs=32, capture_features=False):
    """Logits (and penultimate features = input of model.classifier) in dataset order."""
    feats = []
    handle = None
    if capture_features:
        handle = model.classifier.register_forward_hook(
            lambda m, inp, out: feats.append(inp[0].detach().float().cpu()))
    chunks = []
    for b in range(0, len(texts), bs):
        enc = tok(texts[b:b + bs], truncation=True, max_length=512,
                  padding=True, return_tensors="pt").to(device)
        chunks.append(model(**enc).logits.float().cpu())
    if handle is not None:
        handle.remove()
    logits = torch.cat(chunks).numpy()
    h = torch.cat(feats).numpy() if capture_features else None
    return logits, h


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(C), average="macro", zero_division=0)


def report(rows, tag, y, base_pred, logits=None, pred=None, secs=None, note=""):
    if pred is None:
        pred = logits.argmax(1)
    f1 = mf1(y, pred)
    flips = int((pred != base_pred).sum())
    hist = np.bincount(pred, minlength=C)
    top = hist.max() / hist.sum()
    collapse = "COLLAPSE" if top > 0.60 else ""
    rows.append(dict(arm=tag, macro_f1=round(f1, 4), flips=flips,
                     top_class_frac=round(float(top), 3), secs=None if secs is None else round(secs, 1),
                     note=(note + (" " + collapse if collapse else "")).strip()))
    logger.info(f"{tag:42s} F1={f1:.4f} flips={flips:4d} top={top:.2f} {collapse} {note}")
    return pred


# ---------------------------------------------------------------- TTA machinery
def ln_params(model):
    ps = []
    for m in model.model.modules():          # encoder only (matches E23's 45 affines)
        if isinstance(m, torch.nn.LayerNorm):
            ps += [p for p in m.parameters()]
    for p in ps:
        p.requires_grad_(True)
    return ps


def entropy(logits):
    logp = F.log_softmax(logits, -1)
    return -(logp.exp() * logp).sum(-1)


def adapt_entropy(model, tok, texts, device, lr, n_adapt=2048, bs=8, lam=1.0,
                  mode="shot", rho=0.05, seed=0):
    """One pass of entropy-min adaptation over n_adapt random rows on LN affines.
    mode='shot': IM loss (ent - lam * marginal-ent).  mode='sar': entropy filter
    H<0.4*lnC + SAM(rho) on the filtered entropy (no diversity term, per SAR)."""
    for p in model.parameters():
        p.requires_grad_(False)
    ps = ln_params(model)
    opt = torch.optim.Adam(ps, lr=lr)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(texts))[:n_adapt]
    model.train()  # no dropout anyway (all p=0); LN has no mode-dependent behavior
    e0 = 0.4 * math.log(C)
    ent_start = ent_end = None
    for b in range(0, len(idx), bs):
        rows = [texts[i] for i in idx[b:b + bs]]
        enc = tok(rows, truncation=True, max_length=512,
                  padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        ent = entropy(logits)
        if ent_start is None:
            ent_start = float(ent.mean())
        if mode == "shot":
            marg = logits.softmax(-1).mean(0)
            loss = ent.mean() - lam * (-(marg * marg.clamp_min(1e-8).log()).sum())
            opt.zero_grad(); loss.backward(); opt.step()
        else:  # sar
            mask = ent < e0
            if mask.sum() == 0:
                continue
            loss1 = ent[mask].mean()
            opt.zero_grad(); loss1.backward()
            gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in ps if p.grad is not None))
            eps = []
            with torch.no_grad():
                for p in ps:
                    e = (rho * p.grad / (gnorm + 1e-12)) if p.grad is not None else None
                    eps.append(e)
                    if e is not None:
                        p.add_(e)
            logits2 = model(**enc).logits
            loss2 = entropy(logits2)[mask].mean()   # same filter, per SAR
            opt.zero_grad(); loss2.backward()
            with torch.no_grad():
                for p, e in zip(ps, eps):
                    if e is not None:
                        p.sub_(e)
            opt.step()
        ent_end = float(ent.mean())
    model.eval()
    return ent_start, ent_end


# ---------------------------------------------------------------- T3A / kNN
def t3a_predict(h, logits, W, b, M):
    """Online T3A: normalized features, supports seeded with classifier weights."""
    z = h / np.linalg.norm(h, axis=1, keepdims=True)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    sup = [[Wn[k]] for k in range(C)]              # feature list per class
    sup_ent = [[0.0] for _ in range(C)]            # seed = most confident
    preds = np.empty(len(z), dtype=np.int64)
    for i in range(len(z)):
        cent = np.stack([np.mean(s, 0) for s in sup])
        cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
        scores = cent @ z[i]
        yhat = int(scores.argmax())
        preds[i] = yhat
        p = np.exp(scores - scores.max()); p /= p.sum()
        ent_i = float(-(p * np.log(p + 1e-12)).sum())
        sup[yhat].append(z[i]); sup_ent[yhat].append(ent_i)
        if M > 0 and len(sup[yhat]) > M:
            keep = np.argsort(sup_ent[yhat])[:M]
            sup[yhat] = [sup[yhat][j] for j in keep]
            sup_ent[yhat] = [sup_ent[yhat][j] for j in keep]
    return preds


def knn_propagate(h, probs, k, alpha, steps):
    z = h / np.linalg.norm(h, axis=1, keepdims=True)
    sims = z @ z.T
    np.fill_diagonal(sims, -np.inf)
    nbr = np.argpartition(-sims, k, axis=1)[:, :k]
    p = probs.copy()
    for _ in range(steps):
        p = (1 - alpha) * p + alpha * p[nbr].mean(1)
    return p


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--adapt_bs", type=int, default=8)
    ap.add_argument("--mc_k", type=int, default=10)
    args = ap.parse_args()
    log_cmd()

    device = torch.device("cuda")
    texts, y = load_slice(args.data_dir)
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    rows, preds_store = [], {}

    # ---- base fp32 + features + parity ----
    model = fresh_model(device)
    t0 = time.perf_counter()
    base_logits, h = forward_all(model, tok, texts, device, args.bs, capture_features=True)
    W = model.classifier.weight.detach().float().cpu().numpy()
    bvec = model.classifier.bias.detach().float().cpu().numpy() if model.classifier.bias is not None else np.zeros(C)
    pool = np.load(POOL_NPZ)["logits"]
    argmax_match = (base_logits.argmax(1) == pool.argmax(1)).mean()
    logger.info(f"parity vs pool npz: argmax agreement {argmax_match:.4f}, "
                f"max|dlogit|={np.abs(base_logits - pool).max():.4f}")
    base_pred = report(rows, "base fp32", y, base_logits.argmax(1), logits=base_logits,
                       secs=time.perf_counter() - t0,
                       note=f"parity argmax={argmax_match:.4f}")
    preds_store["base"] = base_pred

    # ---- fp16 arms (fresh load per precision — never re-cast a forwarded model) ----
    del model; torch.cuda.empty_cache()
    model = fresh_model(device, torch.float16)
    t0 = time.perf_counter()
    l16, _ = forward_all(model, tok, texts, device, args.bs)
    report(rows, "fp16 (ship precision)", y, base_pred, logits=l16, secs=time.perf_counter() - t0)
    model.head.float(); model.classifier.float()
    hook = model.head.register_forward_pre_hook(lambda m, inp: (inp[0].float(),))
    t0 = time.perf_counter()
    l16h, _ = forward_all(model, tok, texts, device, args.bs)
    report(rows, "fp16 body + fp32 head", y, base_pred, logits=l16h, secs=time.perf_counter() - t0)
    hook.remove()
    del model; torch.cuda.empty_cache()

    # ---- MC-dropout probe (off-label: trained with p=0) ----
    for p_drop in (0.05, 0.10):
        model = fresh_model(device, torch.float16)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p_drop
        model.train()
        t0 = time.perf_counter()
        acc = np.zeros((len(texts), C), dtype=np.float64)
        for kk in range(args.mc_k):
            torch.manual_seed(1000 + kk)
            with torch.no_grad():
                lk, _ = forward_all(model, tok, texts, device, args.bs)
            acc += torch.softmax(torch.from_numpy(lk), -1).numpy()
        pmc = (acc / args.mc_k)
        pr = report(rows, f"mcdrop p={p_drop} K={args.mc_k} (OFF-LABEL)", y, base_pred,
                    pred=pmc.argmax(1), secs=time.perf_counter() - t0,
                    note="trained w/ dropout 0.0")
        preds_store[f"mcdrop_{p_drop}"] = pr
        del model; torch.cuda.empty_cache()

    # ---- SHOT/IM reference + SAR ----
    for mode, lr in (("shot", 2e-4), ("shot", 1e-3), ("sar", 2e-4), ("sar", 1e-3)):
        model = fresh_model(device)
        t0 = time.perf_counter()
        e_s, e_e = adapt_entropy(model, tok, texts, device, lr,
                                 bs=args.adapt_bs, mode=mode)
        la, _ = forward_all(model, tok, texts, device, args.bs)
        pr = report(rows, f"{mode} lr={lr:g} (LN affines, n=2048)", y, base_pred,
                    logits=la, secs=time.perf_counter() - t0,
                    note=f"ent {e_s:.3f}->{e_e:.3f}")
        preds_store[f"{mode}_{lr:g}"] = pr
        del model; torch.cuda.empty_cache()

    # ---- T3A (backprop-free, on base fp32 features) ----
    for M in (20, 50, 100, 0):
        t0 = time.perf_counter()
        pr = t3a_predict(h, base_logits, W, bvec, M)
        report(rows, f"t3a M={M or 'inf'}", y, base_pred, pred=pr,
               secs=time.perf_counter() - t0)
        preds_store[f"t3a_{M}"] = pr

    # ---- kNN label propagation, honest 2-fold grid pick ----
    probs = torch.softmax(torch.from_numpy(base_logits), -1).numpy()
    grid = [(k, a, s) for k in (5, 10, 20, 50) for a in (0.1, 0.2, 0.3) for s in (1, 2)]
    fa, fb = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=0)
    cache = {g: knn_propagate(h, probs, *g).argmax(1) for g in grid}
    for g in grid:  # transparency: full-slice number per config
        logger.info(f"  knn k={g[0]:2d} a={g[1]} s={g[2]}  full-slice F1={mf1(y, cache[g]):.4f}")
    heldout = []
    for fit_idx, ev_idx in ((fa, fb), (fb, fa)):
        best = max(grid, key=lambda g: mf1(y[fit_idx], cache[g][fit_idx]))
        heldout.append(mf1(y[ev_idx], cache[best][ev_idx]))
        logger.info(f"  fold pick {best} -> held-out F1 {heldout[-1]:.4f}")
    base_half = 0.5 * (mf1(y[fb], base_pred[fb]) + mf1(y[fa], base_pred[fa]))
    rows.append(dict(arm="knn label-prop (honest 2-fold)", macro_f1=round(float(np.mean(heldout)), 4),
                     flips=-1, top_class_frac=-1, secs=None,
                     note=f"vs base half-mean {base_half:.4f}"))
    logger.info(f"knn honest-2-fold F1={np.mean(heldout):.4f} vs base {base_half:.4f}")

    # ---- dump ----
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    np.savez(OUT_NPZ, y=y, **{k: v for k, v in preds_store.items()})
    logger.success(f"E46 done -> {OUT_CSV}")


if __name__ == "__main__":
    main()
