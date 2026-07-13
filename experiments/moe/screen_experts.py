"""E33 follow-up: screen the 16 MoE experts (4 MoEs x 4) as ensemble members,
using the per-expert holdout logprobs already cached in each moe_val_parts.npz.
All 16 experts held out the SAME 3,500 full_data rows (seed-42 split), so this is
honest for expert-only combos. Uniform means only (no fitting — E26/E29: fitted
combos overfit this slice). LB still judges any promoted combo."""
import glob
import numpy as np
from sklearn.metrics import f1_score

CHAMP = 0.7803      # granite champion, same 3,500 holdout
BEST_BLEND = 0.7702  # best single MoE (k6)
TRIO_LB = 0.78719   # context only (different metric/rows)

def mf1(probs, y):
    return f1_score(y, probs.argmax(-1), labels=list(range(14)),
                    average="macro", zero_division=0)

fs = sorted(glob.glob("output/moe_*e33*/moe_val_parts.npz"))
tags = [f.split("e33_moe_")[1].split("/")[0] for f in fs]
data = [np.load(f) for f in fs]

# alignment check: same rows + labels across all 4 MoEs
va0, y0 = data[0]["va_idx"], data[0]["labels"]
for t, d in zip(tags, data):
    assert np.array_equal(d["va_idx"], va0), f"{t} va_idx differs — not the same holdout!"
    assert np.array_equal(d["labels"], y0), f"{t} labels differ!"
y = y0
print(f"aligned: {len(y)} rows, {len(fs)} MoEs -> 16 experts\n")

# per-expert probs -> (16, N, 14) with names
names, P = [], []
for t, d in zip(tags, data):
    p = np.exp(d["expert_logprobs"])       # (N, 4, 14)
    for e in range(4):
        names.append(f"{t}_e{e}")
        P.append(p[:, e])
P = np.stack(P)                            # (16, N, 14)
solo = np.array([mf1(P[i], y) for i in range(16)])

print("solo macro-F1 per expert (sorted):")
order = np.argsort(-solo)
for i in order:
    print(f"  {names[i]:10s} {solo[i]:.4f}")

# --- honest uniform-mean combos (no fitting) ---
def combo(idxs):
    return mf1(P[list(idxs)].mean(0), y)

alive = [i for i in range(16) if solo[i] >= 0.65]      # drop dead/weak experts
strong = [i for i in range(16) if solo[i] >= 0.70]
top1_per_moe = [order[[names[j].startswith(t) for j in order].index(True)]
                for t in tags]  # strongest expert of each MoE (diverse trunks)

print(f"\n--- uniform-mean ensembles (honest, no fitting) ---")
print(f"  best single expert        : {solo.max():.4f}  ({names[order[0]]})")
print(f"  mean ALL 16               : {combo(range(16)):.4f}")
print(f"  mean alive (solo>=.65, n={len(alive)}): {combo(alive):.4f}")
print(f"  mean strong (solo>=.70, n={len(strong)}): {combo(strong):.4f}")
print(f"  mean top-expert/MoE (n=4) : {combo(top1_per_moe):.4f}  [{', '.join(names[i] for i in top1_per_moe)}]")
# top-k strongest overall
for k in (2, 3, 4, 6, 8):
    print(f"  mean top-{k} overall        : {combo(order[:k]):.4f}")

print(f"\n--- reference ---")
print(f"  champion single           : {CHAMP:.4f}")
print(f"  best MoE blend (k6)        : {BEST_BLEND:.4f}")
print(f"  trio LB (diff rows/metric) : {TRIO_LB:.4f}")

# --- error diversity among the strong experts (why/why-not it helps) ---
print(f"\n--- pairwise error correlation among strong experts (n={len(strong)}) ---")
wrong = np.stack([P[i].argmax(-1) != y for i in strong])   # (S, N) bool
S = len(strong)
if S >= 2:
    corr = np.corrcoef(wrong.astype(float))
    iu = np.triu_indices(S, 1)
    print(f"  mean pairwise error-corr : {corr[iu].mean():.3f} "
          f"(lower = more diverse; trio members ~0.5-0.6 for comparison)")
    # cross-MoE vs within-MoE correlation
    same_moe, diff_moe = [], []
    for a in range(S):
        for b in range(a+1, S):
            (same_moe if names[strong[a]].split("_")[0]==names[strong[b]].split("_")[0]
             else diff_moe).append(corr[a, b])
    if same_moe: print(f"  within-MoE pairs (shared trunk): {np.mean(same_moe):.3f}")
    if diff_moe: print(f"  cross-MoE pairs (diff trunk)   : {np.mean(diff_moe):.3f}")
