"""Compare post-hoc logit calibration methods on saved val logits (no training).

Methods: none / bias coordinate-ascent (current shipped recipe) / extended CA /
temperature+bias / vector scaling (diag) / matrix scaling (full 14x14).
Protocol: honest 2-fold (tune on one half of val, score the other, average) +
full-fit (tune and score on all of val — optimistic; what we'd actually ship).

Usage:
  # 1) dump champion val logits once
  python -m analysis.calibration_methods --dump --ckpt submission/model/bge-m3-fp32-backup
  # 2) compare methods
  python -m analysis.calibration_methods
Outputs: analysis/val_logits.npz (step 1), stdout table (step 2).
"""
import argparse
import os

import numpy as np

K = 14
LOGITS_PATH = os.path.join(os.path.dirname(__file__), "val_logits.npz")


def macro_f1(logits, y):
    p = logits.argmax(1)
    f1s = []
    for c in range(K):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def coord_ascent(L, y, T=1.0, rounds=50, grid=None, bias=None):
    """Per-class additive bias maximizing macro-F1, by coordinate ascent."""
    if grid is None:
        grid = np.round(np.arange(-2.0, 2.001, 0.02), 3)
    Ls = L / T
    b = np.zeros(K) if bias is None else bias.copy()
    best = macro_f1(Ls + b, y)
    for _ in range(rounds):
        improved = False
        for c in range(K):
            cur, cbest = b[c], best
            for g in grid:
                b[c] = g
                f = macro_f1(Ls + b, y)
                if f > cbest:
                    cbest, cur = f, g
            b[c] = cur
            if cbest > best:
                best, improved = cbest, True
        if not improved:
            break
    return b, best


def fit_vector_scaling(L, y, full=False, C=1.0):
    """NLL-fitted rescaling: diag (per-class w,b) or full 14x14 logistic regression."""
    if full:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, C=C)
        m.fit(L, y)
        return lambda X: X @ m.coef_.T + m.intercept_
    from scipy.optimize import minimize

    def nll(params):
        w, b = params[:K], params[K:]
        z = L * w + b
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()

    res = minimize(nll, np.concatenate([np.ones(K), np.zeros(K)]), method="L-BFGS-B")
    w, b = res.x[:K], res.x[K:]
    return lambda X: X * w + b


def dump_logits(ckpt, data_dir, max_len, out_path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from src.data import CLASS_TO_ID, load_samples, serialize, split_indices

    samples, y = load_samples(data_dir)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    _, va = split_indices(y, seed=42)
    texts = [serialize(samples[i]) for i in va]
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, local_files_only=True, torch_dtype=torch.float16).cuda().eval()
    coll = DataCollatorWithPadding(tokenizer=tok)
    encs = []
    for s in range(0, len(texts), 1024):
        e = tok(texts[s:s + 1024], truncation=True, max_length=max_len, padding=False)
        encs.extend({"input_ids": i, "attention_mask": a}
                    for i, a in zip(e["input_ids"], e["attention_mask"]))
    order = sorted(range(len(encs)), key=lambda i: len(encs[i]["input_ids"]), reverse=True)
    logits = np.zeros((len(encs), K), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(order), 96):
            idx = order[s:s + 96]
            b = {k: v.cuda() for k, v in coll([encs[i] for i in idx]).items()}
            out = model(**b).logits.float().cpu().numpy()
            for j, i in enumerate(idx):
                logits[i] = out[j]
    np.savez(out_path, logits=logits, labels=y_ids[va], va=va)
    print(f"saved {logits.shape} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="compute+save val logits first")
    ap.add_argument("--ckpt", default="submission/model/bge-m3-fp32-backup")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--logits", default=LOGITS_PATH)
    args = ap.parse_args()

    if args.dump:
        dump_logits(args.ckpt, args.data_dir, args.max_len, args.logits)
        return

    d = np.load(args.logits)
    L, Y = d["logits"].astype(np.float64), d["labels"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y))
    folds = [perm[: len(Y) // 2], perm[len(Y) // 2:]]

    def evaluate(name, fit_fn):
        hon = []
        for a, b in [(0, 1), (1, 0)]:
            tf = fit_fn(L[folds[a]], Y[folds[a]])
            hon.append(macro_f1(tf(L[folds[b]]), Y[folds[b]]))
        tf_full = fit_fn(L, Y)
        # hon[0] = tuned on A, scored on B; hon[1] = tuned on B, scored on A
        return name, hon[0], hon[1], float(np.mean(hon)), macro_f1(tf_full(L), Y)

    coarse = np.round(np.arange(-2, 2.001, 0.05), 2)
    base_a = macro_f1(L[folds[1]], Y[folds[1]])   # "scored on B" (no tuning involved)
    base_b = macro_f1(L[folds[0]], Y[folds[0]])   # "scored on A"
    results = [("no calibration (argmax)", base_a, base_b,
                float(np.mean([base_a, base_b])), macro_f1(L, Y))]

    results.append(evaluate("bias CA (current: 3 rounds, step .05)",
        lambda Lt, yt: (lambda b: lambda X: X + b)(coord_ascent(Lt, yt, rounds=3, grid=coarse)[0])))
    results.append(evaluate("bias CA extended (50 rounds, step .02)",
        lambda Lt, yt: (lambda b: lambda X: X + b)(coord_ascent(Lt, yt, rounds=50)[0])))

    def temp_bias(Lt, yt):
        best = (1.0, -1.0, None)
        for T in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            b, f = coord_ascent(Lt, yt, T=T, rounds=6, grid=coarse)
            if f > best[1]:
                best = (T, f, b)
        T, _, b = best
        return lambda X: X / T + b
    results.append(evaluate("temperature + bias CA", temp_bias))

    def vector_scaling(Lt, yt):
        tf = fit_vector_scaling(Lt, yt, full=False)
        b, _ = coord_ascent(tf(Lt), yt, rounds=6, grid=coarse)
        return lambda X: tf(X) + b
    results.append(evaluate("vector scaling (diag NLL) + bias CA", vector_scaling))

    def matrix_scaling(Lt, yt):
        tf = fit_vector_scaling(Lt, yt, full=True, C=0.1)
        b, _ = coord_ascent(tf(Lt), yt, rounds=6, grid=coarse)
        return lambda X: tf(X) + b
    results.append(evaluate("matrix scaling (14x14, C=0.1) + bias CA", matrix_scaling))

    base = results[0]
    print(f"\n{'method':45s} {'tuneA->B':>19s} {'tuneB->A':>19s} {'mean':>19s} {'full-fit':>19s}")
    for n, ha, hb, hm, f in results:
        cells = []
        for v, b in zip((ha, hb, hm, f), base[1:]):
            d = v - b
            cells.append(f"{v:.4f} ({d:+.4f})" if n != base[0] else f"{v:.4f}")
        print(f"{n:45s} {cells[0]:>19s} {cells[1]:>19s} {cells[2]:>19s} {cells[3]:>19s}")


if __name__ == "__main__":
    main()
