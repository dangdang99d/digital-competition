"""E30 phase 0 — session-grouped OOF harvest for the trio member recipes.

For (member, fold): train the member's EXACT recipe from scratch with
`--session_fold fold --session_splits 5` (StratifiedGroupKFold by session over the
FULL 70k, leakage asserted 0 inside finetune.py), then run the fold-model over its
held-out fold and save raw fp32 probs + penultimate embeddings (input of the final
classifier Linear, captured via forward hook — architecture-agnostic).

Per-fold caches  analysis/cache/e30_oof_<member>_f<fold>.npz  {probs, emb, rows}
Merged           analysis/cache/e30_oof_<member>.npz          {probs(70k,14) fp32,
                 emb(70k,H) fp16, covered(70k) bool} — load_samples order.

Usage — 15 (member, fold) units, run on vast.ai per VAST.md (pinned template image,
DONE-sentinel; user drives instances). One unit per GPU:
  python experiments/ensemble/oof_harvest.py --member is3 --fold 0        # train+harvest
  python experiments/ensemble/oof_harvest.py --member is3 --merge         # after 5 folds
  python experiments/ensemble/oof_harvest.py --dry_run                    # print all 15 cmds
After all merges: python experiments/ensemble/build_oof_teacher.py

⚠️ VERIFY BEFORE LAUNCH (dry_run prints these): member arg lists below were
reconstructed from ft_results CSVs + experiment reports; diff against the original
run logs (is3: --init_seed 3 assumption; aum06: extended fulldata keep-set; e25c:
bf16+warmup 0.1 — bf16 is fine for TRAINING boxes, not the T4).
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.getcwd())
# numpy / src.data are imported lazily inside harvest()/merge() so --dry_run and the
# training-launch path work on boxes without the dacon env.

CACHE = "analysis/cache"
N_SPLITS = 5
COMMON = ("--model ibm-granite/granite-embedding-311m-multilingual-r2 "
          "--epochs 3 --lr 2e-5 --batch_size 16 --grad_accum 1 --group_by_length "
          "--max_len 512 --seed 42 --loss ls "
          "--out_dir ./output/pat --results_name ft_results_e30_oof.csv")
MEMBERS = {  # trio member recipes (LB 0.78719) — verify vs original logs before launch
    "is3":   {"serialize": "richargs", "extra": "--init_seed 3"},
    "aum06": {"serialize": "richargs",
              "extra": "--keep_indices experiments/coreset/keepsets/aum_drop06_fulldata.npy"},
    "e25c":  {"serialize": "richmeta", "extra": "--precision bf16 --warmup_ratio 0.1"},
}


def train_cmd(member, fold):
    m = MEMBERS[member]
    tag = f"e30_{member}_f{fold}"
    return (f"{sys.executable} -u -m src.finetune --tag {tag} "
            f"--serialize {m['serialize']} {m['extra']} {COMMON} "
            f"--session_fold {fold} --session_splits {N_SPLITS}"), tag


def resolve_ckpt(run_dir):
    import glob
    if os.path.exists(os.path.join(run_dir, "model.safetensors")):
        return run_dir
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    best = [c for c in cks if os.path.exists(os.path.join(c, "model.safetensors"))]
    assert best, f"no checkpoint under {run_dir}"
    return best[-1]


def harvest(member, fold, batch_size=64):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.data import ALL_CLASSES, CLASS_TO_ID, build_texts, load_samples, \
        session_fold_indices
    os.makedirs(CACHE, exist_ok=True)

    samples, labels = load_samples("./data")
    y = np.array([CLASS_TO_ID[a] for a in labels])
    _, va = session_fold_indices(samples, y, fold, n_splits=N_SPLITS, seed=42)
    texts = build_texts(samples, variant=MEMBERS[member]["serialize"])
    txv = [texts[i] for i in va]

    run_dir = None
    import glob
    hits = glob.glob(f"output/pat/ft_*_e30_{member}_f{fold}")
    assert len(hits) == 1, f"expected exactly one run dir for e30_{member}_f{fold}: {hits}"
    run_dir = hits[0]
    ck = resolve_ckpt(run_dir)
    tok = AutoTokenizer.from_pretrained(run_dir, trust_remote_code=True)
    tok.truncation_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        ck, torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()

    # penultimate = input of the last Linear with out_features == n_classes
    head = [m for m in model.modules()
            if isinstance(m, torch.nn.Linear) and m.out_features == len(ALL_CLASSES)][-1]
    grabbed = {}
    head.register_forward_hook(lambda mod, inp, out: grabbed.update(h=inp[0].detach()))

    probs, embs = [], []
    with torch.no_grad():
        for b in range(0, len(txv), batch_size):
            enc = tok(txv[b:b + batch_size], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to("cuda")
            logits = model(**enc).logits.float()
            probs.append(torch.softmax(logits, -1).cpu().numpy())
            embs.append(grabbed["h"].float().cpu().numpy().astype(np.float16))
    out = f"{CACHE}/e30_oof_{member}_f{fold}.npz"
    np.savez(out, probs=np.concatenate(probs), emb=np.concatenate(embs),
             rows=va.astype(np.int32))
    print(f"harvested fold {fold}: {len(va)} rows -> {out}")


def merge(member):
    import numpy as np

    from src.data import load_samples

    os.makedirs(CACHE, exist_ok=True)
    samples, labels = load_samples("./data")
    n = len(samples)
    probs = emb = None
    covered = np.zeros(n, bool)
    for f in range(N_SPLITS):
        D = np.load(f"{CACHE}/e30_oof_{member}_f{f}.npz")
        if probs is None:
            probs = np.zeros((n, D["probs"].shape[1]), np.float32)
            emb = np.zeros((n, D["emb"].shape[1]), np.float16)
        rows = D["rows"]
        assert not covered[rows].any(), f"fold {f}: overlapping rows — folds inconsistent"
        probs[rows], emb[rows], covered[rows] = D["probs"], D["emb"], True
    assert covered.all(), f"coverage {covered.sum()}/{n} — missing fold caches"
    out = f"{CACHE}/e30_oof_{member}.npz"
    np.savez(out, probs=probs, emb=emb, covered=covered)
    print(f"merged {member}: {n} rows, emb dim {emb.shape[1]} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", choices=list(MEMBERS))
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_train", action="store_true",
                    help="harvest only (fold-model already trained)")
    args = ap.parse_args()

    if args.dry_run:
        for m in MEMBERS:
            for f in range(N_SPLITS):
                print(train_cmd(m, f)[0])
        return
    assert args.member, "--member required"
    if args.merge:
        merge(args.member)
        return
    assert 0 <= args.fold < N_SPLITS, "--fold 0..4 required"
    if not args.skip_train:
        cmd, tag = train_cmd(args.member, args.fold)
        print(f"TRAIN {tag}: {cmd}", flush=True)
        subprocess.run(cmd, shell=True, check=True)
    harvest(args.member, args.fold)


if __name__ == "__main__":
    main()
