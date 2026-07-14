# Depth pruning (structured) — drop whole transformer layers + recovery FT

**Method:** ShortGPT/BI layer selection (Block Influence = cosine redundancy) → remove the
lowest-influence layers → recovery fine-tune. Structured: output is a smaller *dense* model,
speedup unconditional (~linear in layers removed). Index: [results.md](results.md).
Tools: `experiments/compression/bi_probe.py` (BI, train rows only) ·
`experiments/compression/eval_compress.py` (3-metric eval on the honest 3.5k, `--drop_layers`
zero-shot, `--remap` for vocab-pruned ckpts) · `--keep_layer_idx`+`--init_from`(+
`--remap_tokens`) in finetune.py for recovery.

**Baselines (honest 3.5k held-out, fp16-eager eval config):** granite t031 **0.78578** ·
qwen3_ls **0.76569** (trainer-metric parity: 0.7822-config / 0.7659 full_data CV ✓).

## Two method-level lessons (2026-07-14, load-bearing)

1. **BI prune-order is CHECKPOINT-specific.** t031's order applied to a *sibling* AWP granite
   = −0.103 @keep-18; that model's own order = −0.041. Same backbone, same recipe family —
   still not transferable. Always re-probe the exact checkpoint you prune.
2. **Zero-shot damage ≠ post-recovery verdict — recovery power is the story.** qwen3 keep-14
   zero-shot is **−0.37…−0.40** (catastrophic), yet E16's keep-14 **+ recovery landed at
   −0.0005**. Recovery-FT closed a ~0.40 hole on the decoder stack. Corollary: in-sample
   zero-shot probes also understate damage (t031 in-sample keep-18 read −0.015; honest-slice
   sibling equivalent −0.04…−0.10) — probe on the 3.5k held-out, judge after recovery.

## granite track (E31) — baseline `submit_0714_awp_t031` (LB 0.79300)

**BI (own, 1024 train rows):** ascending order `[12,4,11,13,3,2,16,17,6,7,…]` — mid-stack
{11,12,13} + early {2,3,4} redundant; **L18, L15, L21, L0 load-bearing** (encoder profile;
reproduces the local-box probe on the same checkpoint ✓).

**Zero-shot layer-drop, honest 3.5k (base 0.78578) — 3-metric protocol:**

| kept | dropped (own BI) | ΔF1 | flip-rate | \|Δlogit\| mean | KL |
|---|---|---|---|---|---|
| 20 | {12,4} | **−0.0075** | 1.6% | 0.049 | 0.005 |
| 18 | +{11,13} | **−0.0200** | 4.8% | 0.188 | 0.057 |
| 16 | +{3,2} | −0.0386 | 6.5% | 0.198 | 0.085 |

Zero-shot damage is gentle through keep-16.
Note t031 zero-shot-prunes markedly better than the sibling g_awp_swa (−0.020 vs −0.041
@keep-18, each with its own BI) — more per-checkpoint variance.

**Recovery arms DONE (2ep lr1e-5 LS from t031; all peaked at ep1, ep2 declined; same
eval config as zero-shot, honest 3.5k):**

| arm | zero-shot | post-recovery | recovery uplift | net vs 0.78578 | flip | KL |
|---|---|---|---|---|---|---|
| keep-20 | −0.0075 | 0.77387 | **−0.004 — recovery HURT** | −0.0119 | 6.2% | 0.020 |
| keep-18 | −0.0200 | 0.78064 | +0.0149 | **−0.0051** | 5.5% | 0.019 |
| keep-16 | −0.0386 | 0.77693 | +0.0297 | −0.0089 | 5.8% | 0.021 |

**granite depth verdict (2ep recovery):** best net = keep-18 −0.0051; none clears the
−0.002 Stage-A gate yet. Recovery uplift GROWS with damage (−0.004 → +0.030), and for the
mild keep-20 cut recovery is NET-NEGATIVE — the E24 warm-start degradation outweighs healing
when there's little to heal (ship mild prunes zero-shot; recover deep ones). All arms peaked
at ep1 → try 1ep / lower-lr / AWP-recipe recovery before closing the axis. Speed at keep-18
≈ −18% encoder wall-clock.

## ⚠️ granite reload trap — ModernBERT index-wired attention (found 2026-07-14)

ModernBERT assigns **global vs local-128 attention (and rope theta) by layer index at
`__init__`** (`layer_id % global_attn_every_n_layers`). Depth-pruning renumbers layers, so a
plain `from_pretrained` of a pruned ckpt **rebuilds the wrong attention pattern** — silently:
t031 keep-20/18 arms trained to 0.774/0.781 in-process but reloaded at 0.405/0.345. keep-16
survived only because dropping consecutive triples ({2,3,4},{11,12,13}) preserves index%3.
- **Fix (in repo):** `prune_layers` now stamps `config.kept_layer_indices` +
  `pruned_from_depth`; `eval_compress.py` reconstruct-loads (build full-depth → prune to the
  original kept indices → load weights). **Any shipped depth-pruned granite needs the same
  reconstruct-load in its `script.py`** — or restrict keep-sets to mod-3-preserving drops
  (drop consecutive triples), which stay standard-loadable.
- qwen3 is immune (uniform attention every layer) — why E16 never hit this.

**Sibling pilot (g_awp_swa, slice-0.7822 AWP granite — NOT the baseline; recovery-uplift
shape only), DONE 2026-07-14** (2ep lr1e-5 LS; zs = fp16-eager config, recovery = trainer
metric — small config offset, don't over-read second decimals):

| arm | zero-shot | post-recovery (best ep) | recovery uplift | net vs 0.7822 |
|---|---|---|---|---|
| keep-18 (own BI) | 0.7397 (−0.041) | **0.7672** (ep1; ep2 declined 0.7649) | **+0.028** | −0.015 |
| keep-16 (own BI) | 0.5788 (−0.202) | **0.7728** (ep2) | **+0.194** | **−0.009** |

**Pilot read-out: granite recovery ALSO heals massively** — keep-16 closed a −0.20 hole to
−0.009, and post-recovery keep-16 ≥ keep-18 (deeper cut, better final — zero-shot ordering
did not survive recovery). Both nets still miss the −0.002 Stage-A gate at 2ep; whether more
epochs/AWP-recovery closes the rest → t031 arms (the real baseline) decide.

## granite — earlier local probes (E31 A1, 2026-07-14, local 4060, ⚠️ in-sample slice)

Ran on the **t031 champion** (granite-311m E8a+LS+AWP, LB-0.79300 single), 1024 samples, fp16.
**Refutes the false E16b "granite can't depth-prune"** (that was a 1-layer-model bug): granite
has real but *modest*, non-monotonic layer redundancy — NOT the half-depth-free profile qwen3 had.

⚠️ **Slice caveat:** this used a session-grouped fold, which is **in-sample** for the
full_data t031 → the absolute F1 (0.787/0.803) is inflated (LB is 0.793). The pruning **deltas**
are roughly valid (same rows both sides), but **retrofit onto the 3.5k held-out (`va_eval`)**
and add the prediction-drift + logit-change metrics — see [results.md](results.md) protocol.

**Block-Influence** `BI_i = 1 − mean_token cos(h_in, h_out)`:
- Load-bearing (high BI, keep): **L18 0.160 · L15 0.125 · L21 0.125 · L0 0.091**.
- Most redundant (low BI, prune first): **L12 0.025 · L4 0.025 · L11 0.026 · L13 0.026 ·
  L3 0.030 · L2 0.031** — a mid-stack (L11–13) + early-mid (L2–4) cluster.
- Ascending-BI order: `[12,4,11,13,3,2,16,6,17,7,...]`.

**Zero-shot layer-drop** (identity pass-through; kept layers retain index + ModernBERT
global/local pattern; NO recovery) — raw damage recovery-FT must close (base 0.78743 @1024):

| kept | dropped | macro-F1 | Δ |
|---|---|---|---|
| 22 (full) | — | 0.78743 | — |
| 20 | {4,12} | 0.78462 | **−0.0028** (near-free) |
| 18 | {4,11,12,13} | 0.77198 | **−0.0154** (recovery target) |
| 16 | +{2,3} | 0.74900 | −0.0384 (borderline) |
| 14 | +{6,16} | 0.64429 | **−0.1431** (cliff) |
| 11 | +{7,14,17} | 0.31558 | −0.4718 (~random) |

**Verdict (probe-level):** viable but only MODEST. keep-20 nearly free before recovery;
keep-18 (−0.015) is the realistic recovery-FT target (recovery typically closes ~1–2pt of
zero-shot loss → plausibly net-≈0); keep-16 borderline; **keep-14 is a cliff** — half-depth
does NOT transfer from qwen3 (E16 was free at 28→14 *because* decoder stacks are far more
redundant than this encoder — now quantified). Speed payoff modest: drop 4/22 ≈ 18% fewer
layers (~5:06 → ~4:10), NOT qwen3's 2×.

## qwen3 track (E45) — baseline `submit_0707_qwen3_ls` (LB 0.77921)

**BI (own, 1024 train rows):** ascending order `[25,24,13,12,23,14,15,16,22,8,11,26,9,20,…]`
— mid/late stack redundant; **L0, L1, L27 load-bearing** (classic decoder profile).

**Zero-shot layer-drop, honest 3.5k (base 0.76569) — 3-metric protocol:**

| kept | dropped (own BI) | ΔF1 | flip-rate | KL |
|---|---|---|---|---|
| 26 | {25,24} | **−0.006** | 2.5% | 0.052 |
| 24 | +{13,12} | **−0.019** | 5.6% | 0.075 |
| 20 | +{23,14,15,16} | −0.119 | 19.0% | 0.317 |
| 16 | +{22,8,11,26} | −0.319 | 42.9% | 0.989 |
| 14 (own) | +{9,20} | −0.367 | 50.6% | 1.299 |
| 14 (E16 set) | drop `{8,12-18,20,22-26}` | −0.404 | 53.5% | 1.056 |

Zero-shot knee after keep-24. BUT per lesson 2 above, E16 proved recovery closes even the
keep-14 hole (0.7638 final on E8b lineage, ~2× speed) — so recovery arms decide, not this
curve. Recovery on the vocab-pruned qwen3_ls ckpt = `--init_from <zip model> --remap_tokens
<remap.npy> --keep_layer_idx ...` (hook added to finetune.py 2026-07-14). Priority arms:
keep-24 (cheap win?) · keep-20 · keep-14own (re-test E16's full heal under the LS recipe).
Stacks (untested) with FFN low-rank → [low-rank-factorization.md](low-rank-factorization.md).

## Next
- **granite recovery-FT** of keep-20/18/16 vs a from-scratch anchor (E24 warm-start trap) on
  a real GPU box — the actual verdict on whether the keep-18 −0.015 closes. `--keep_layers`
  (NOT `--keep_layer_idx` — the E16b bug). Probes only on the 8GB local card.
- Scripts: `trt_int8_work/bi_probe.py`, `trt_int8_work/zero_shot_drop.py` (gitignored workdir).

## Scope caveats
- BI *predicts* prunability; prune+recovery is the real test.
- ShortGPT block-influence is formula-level here; official repo `icip-cas/ShortGPT` not
  cloned/diffed. Zero-shot drop uses identity pass-through (faithful to residual skip).
