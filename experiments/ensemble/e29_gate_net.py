"""E29 arm ④ — gated fusion: per-row gating net with EMBEDDING input (user 2026-07-12).

Input per row: concatenated member penultimate embeddings (3×768, standardized on the
train fold). Output: 3 member weights (softmax). Prediction: weighted mean of the members'
OOF probability vectors. Trained end-to-end through the blend with NLL; the constraint —
the gate can only BLEND member opinions, never emit its own class — is the overfitting
armor arm ② lacks.

Eval: 5-fold session-grouped CV (STRING labels — the E30 fold convention) vs the uniform
mean null. Gate to matter: ≥ +0.002 ΔmF1.

  python experiments/ensemble/e29_gate_net.py [--epochs 6] [--hidden 0] [--device cpu]
"""
import argparse
import re

import numpy as np

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
MEMBERS = ["is3", "aum06", "e25c"]


def mf1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


def main():
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear gate; else MLP hidden dim")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--input", choices=["emb", "probs"], default="emb")
    ap.add_argument("--objective", choices=["nll", "softf1"], default="nll")
    args = ap.parse_args()
    torch.manual_seed(42)

    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.data import CLASS_TO_ID, load_samples, session_fold_indices
    samples, labels = load_samples("./data")
    y = np.array([CLASS_TO_ID[a] for a in labels])

    caches = {m: np.load(f"analysis/cache/e30_oof_{m}.npz") for m in MEMBERS}
    key = "emb" if args.input == "emb" else "probs"
    E = np.concatenate([caches[m][key].astype(np.float32) for m in MEMBERS], 1)  # N x (2304|42)
    P = np.stack([caches[m]["probs"].astype(np.float32) for m in MEMBERS], 1)       # N x 3 x 14
    uni_pred = P.mean(1).argmax(1)
    print(f"n={len(y)} · uniform null mF1 {mf1(y, uni_pred):.4f}")

    dev = torch.device(args.device)
    hs, ns = [], []
    for f in range(5):
        tr, te = session_fold_indices(samples, labels, f, n_splits=5, seed=42)
        mu, sd = E[tr].mean(0, keepdims=True), E[tr].std(0, keepdims=True) + 1e-6
        Xtr = torch.tensor((E[tr] - mu) / sd, device=dev)
        Xte = torch.tensor((E[te] - mu) / sd, device=dev)
        Ptr = torch.tensor(P[tr], device=dev)
        Pte = torch.tensor(P[te], device=dev)
        ytr = torch.tensor(y[tr], device=dev)

        if args.hidden:
            gate = nn.Sequential(nn.Linear(E.shape[1], args.hidden), nn.ReLU(),
                                 nn.Linear(args.hidden, len(MEMBERS))).to(dev)
        else:
            gate = nn.Linear(E.shape[1], len(MEMBERS)).to(dev)
        opt = torch.optim.Adam(gate.parameters(), lr=args.lr)

        for ep in range(args.epochs):
            perm = torch.randperm(len(ytr), device=dev)
            tot = 0.0
            for b in range(0, len(ytr), args.batch):
                idx = perm[b:b + args.batch]
                w = torch.softmax(gate(Xtr[idx]), -1)                     # B x 3
                blend = (w.unsqueeze(-1) * Ptr[idx]).sum(1).clamp_min(1e-9)
                if args.objective == "softf1":
                    y1 = torch.nn.functional.one_hot(ytr[idx], blend.shape[1]).float()
                    tp = (blend * y1).sum(0)
                    fp = (blend * (1 - y1)).sum(0)
                    fn = ((1 - blend) * y1).sum(0)
                    loss = 1 - (2 * tp / (2 * tp + fp + fn + 1e-9)).mean()
                else:
                    loss = -torch.log(blend[torch.arange(len(idx)), ytr[idx]]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(idx)
            print(f"fold {f} ep {ep}: train NLL {tot / len(ytr):.4f}", flush=True)

        with torch.no_grad():
            w = torch.softmax(gate(Xte), -1)
            pred = (w.unsqueeze(-1) * Pte).sum(1).argmax(-1).cpu().numpy()
            wbar = w.mean(0).cpu().numpy()
        h, n_ = mf1(y[te], pred), mf1(y[te], uni_pred[te])
        hs.append(h); ns.append(n_)
        print(f"fold {f}: gate mF1 {h:.4f} · null {n_:.4f} · Δ {h - n_:+.4f} · "
              f"mean weights {np.round(wbar, 3)}", flush=True)

    d = np.array(hs) - np.array(ns)
    print(f"\nARM[{args.input}/{args.objective}]: gate {np.mean(hs):.4f} vs null {np.mean(ns):.4f} · "
          f"Δ {np.mean(d):+.4f} · folds>0 {(d > 0).sum()}/5 · "
          f"GATE {'PASS' if np.mean(d) >= 0.002 else 'FAIL'}")


if __name__ == "__main__":
    main()
