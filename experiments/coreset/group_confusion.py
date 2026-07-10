"""group_confusion.py — group-level confusion of val predictions (user ask 2026-07-09).

Hypothesis behind a group-routing lever: the 14 classes form 4 confusion groups
(src/data.ACTION_GROUPS: explore/edit/execute/noncode) and even when the class
prediction is wrong, the true label usually sits in the predicted GROUP. Before
proceeding we need, per model: group-level one-vs-rest TP/FP/TN/FN, class vs
group accuracy, and the containment rate P(true group == pred group | class wrong).

Pure CPU re-read of cached val logits — qwen3 champion ruler
(analysis/cache/qwen3_val_logits.npz) + the §3 gate-analysis models
(analysis/cache/coreset_gate_logits.npz). Raw logits, NO calibration.

  PYTHONPATH=/home/ocean/dacon \
    /home/ocean/miniconda3/envs/dacon/bin/python experiments/coreset/group_confusion.py
"""
import numpy as np

from src.data import ACTION_GROUPS, CLASS_TO_ID, GROUP_ID, load_samples, split_indices

GATE_CACHE = "analysis/cache/coreset_gate_logits.npz"
RULER = "analysis/cache/qwen3_val_logits.npz"
GROUPS = list(ACTION_GROUPS)


def main():
    samples, y = load_samples("./data")
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    _, va = split_indices(y, seed=42)
    yv = y_ids[va]
    gid = np.array(GROUP_ID)
    gv = gid[yv]                                   # true group per val row
    N = len(yv)

    models = {"qwen3 (champion ruler)": np.load(RULER)["logits"]}
    models.update(dict(np.load(GATE_CACHE)))

    print(f"val rows {N} · groups {GROUPS} · true-group counts "
          f"{[int((gv == k).sum()) for k in range(len(GROUPS))]}\n")

    print("## Class vs group accuracy — is the wrong answer still in the right group?\n")
    print("| model | class acc | group acc | class-err rate | containment "
          "P(group ok \\| class wrong) |")
    print("|---|---|---|---|---|")
    preds = {}
    for name, z in models.items():
        pred = z.argmax(1)
        preds[name] = pred
        gp = gid[pred]
        wrong = pred != yv
        contain = float((gp[wrong] == gv[wrong]).mean())
        print(f"| {name} | {float((~wrong).mean()):.4f} | {float((gp == gv).mean()):.4f} | "
              f"{float(wrong.mean()):.4f} | {contain:.4f} |")

    print("\n## Group-level one-vs-rest TP/FP/TN/FN (predicted group vs true group)\n")
    print("| model | group | TP | FP | TN | FN | precision | recall | ovr acc |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, pred in preds.items():
        gp = gid[pred]
        for k, gname in enumerate(GROUPS):
            tp = int(((gp == k) & (gv == k)).sum())
            fp = int(((gp == k) & (gv != k)).sum())
            fn = int(((gp != k) & (gv == k)).sum())
            tn = N - tp - fp - fn
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            print(f"| {name} | {gname} | {tp} | {fp} | {tn} | {fn} | "
                  f"{prec:.4f} | {rec:.4f} | {(tp + tn) / N:.4f} |")

    print("\n## 4×4 group confusion (rows = true, cols = predicted), per model\n")
    for name, pred in preds.items():
        gp = gid[pred]
        print(f"**{name}**\n")
        print("| true \\ pred | " + " | ".join(GROUPS) + " |")
        print("|---|" + "---|" * len(GROUPS))
        for k, gname in enumerate(GROUPS):
            row = [int(((gv == k) & (gp == j)).sum()) for j in range(len(GROUPS))]
            print(f"| {gname} | " + " | ".join(str(v) for v in row) + " |")
        print()


if __name__ == "__main__":
    main()
