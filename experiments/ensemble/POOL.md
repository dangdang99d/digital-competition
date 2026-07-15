# Ensemble logit pool — 3.5k held-out slice

Data lives in [`logits/`](logits/): per-model raw fp32 val logits on the **shared 3.5k held-out
slice**, one `.npz` per model, so any ensemble combination can be scored offline (no GPU) via
[`search_logits.py`](search_logits.py). Built by [`harvest_logits.py`](harvest_logits.py)
(split existing caches + GPU-harvest every untested model) and
[`recover_logits.py`](recover_logits.py) (models whose dir-tag had no `ft_results` row).

## The slice
```
_, va       = split_indices(labels, seed=42)                    # 14k
_, va_eval  = train_test_split(va, 0.25, stratify, rs=42)       # 3.5k   (va_eval ⊂ va)
```
Byte-identical to `finetune.py --full_data` / `screen_ensemble.py`. **Any model trained on the
seed-42 split (standard OR full_data) held this slice out → clean.** Verified: the existing
caches' rows and the MoE `va_idx` all equal this `va_eval`.

## Files (in `logits/`)
- `logits/_meta.npz` — `va_idx (3500,)`, `labels (3500,)`, `classes (14,)` — shared row order + gold
- `logits/<tag>.npz` — `logits (3500,14)` float32, rows aligned to `_meta`
- `logits/manifest.jsonl` — every candidate model + decision (`ok`/`skip_*`/`drop_lowf1`/`error`) + self-F1 + serialize + source

## What's in the pool: **85 models** (all self-F1 ≥ 0.70)
Includes families that had **never** been ensemble-tested before this pass:
**E38 AWP-optuna** (`e38_t001fd`/`t031fd`/`t070fd` ≈ 0.786 = top singles; `t031fd` = LB 0.79300 SOTA
single) + E38 ELR+AWP (`e38_t*_elr`), **SWA** (`g_awp_swa` 0.781), E35 loss sweep (`e35_ce/ls/gce/sce/boot`),
E34 adversarial (`e34_c_awp` 0.7808), E30/E26 distillation students, `e32_b1_hdrop`, `e28_full_t019`,
`soup_granite_ls`. Plus the whole prior E26 pool, E28 full-data winners, coreset, E9 losses, TAPT, MoE, qwen3.

## What was excluded (and why)
| decision | n | reason |
|---|---|---|
| `skip_leak` | 17 | did **not** hold out the seed-42 slice — E30 KFold folds (`e30_*_f{0..4}`), `e12_*_s43/s44`. Score high *because in-sample*; F1 gate can't catch this, so excluded by metadata. |
| `skip_incompat` | 13 | arch-modified / non-classifier — `a24_attn`, `a24_sal`, `b24_*`, `e4_ffn`, MLM/`smoke` checkpoints. Load as garbage. |
| `drop_lowf1` | 12 | loaded fine but self-F1 < 0.70 (weak specialists, broken depth-prune caches, weak MoE experts). |
| `skip_no_serialize` | 17 | dir-tag had no `ft_results` serialize row — refused to guess (wrong variant silently poisons). High-value ones (`soup`, `e28_full_t019`, `e26s_*`) recovered with explicit richargs; TAPT-MLM/base-qwen/bge-m3 specialists left out. |
| `error` | — | `hardex` (bge-m3, load fail); **`qwen3_champion_pruned`** needs the remap.npy input path (pruned vocab → full tokenizer emits out-of-range IDs → CUDA assert). Its unpruned sources `e8b_qwen3_richargs_full` / `e8b_ls_qwen3_richargs_full` are already in the pool. |

## Caveat
The 3.5k local slice **under-predicts and mis-ranks ensembles** vs. the DACON LB (E26). Use this
pool to *shortlist* combinations; confirm the finalists on LB.

## Model coverage
[`MODELS.md`](MODELS.md) — added vs excluded, per experiment (regenerate with
`python experiments/ensemble/gen_pool_readme.py`, writes here). All E1–E34 accuracy experiments are in;
only arch/compression experiments (E4/E5/E18, a24/b24 prunes) and leakers (E30 folds, e12_s43/44)
are out.

## Usage
```bash
# quick score search (singles/pairs/triples/greedy)
PYTHONPATH=. python experiments/ensemble/search_logits.py --greedy_len 6 --topk 20 --triples
# robust selection — method A (bootstrap-bagged) + C (diversity), with bootstrap CIs
PYTHONPATH=. python experiments/ensemble/select_ensemble.py --B 1000 --greedy_len 3 --lam 0.05
```

## ⚠️ The slice cannot rank ensembles
The champion trio (real **LB 0.78780**) scores **0.7770 on this slice — the *lowest* of all
candidate trios**, while the highest slice-scoring trio (0.789) is LB-unproven. Method A's
bootstrap shows no model survives even 50% of resamples → the "best combination" is a draw
artifact. **Use this pool to shortlist robust members; decide finalists on LB.**
