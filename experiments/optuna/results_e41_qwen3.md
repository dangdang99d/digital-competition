# E41 · qwen3 AWP hyperparameter search (Optuna) — cross-backbone diversity — `research/optuna`

**Status: ✅ DONE (2026-07-14). 8×5090, 14/16 trials completed (2 killed for the ep6 diagnostic);
best t011 = 0.7776 fold-0 — qwen3 ceiling ~0.003 below granite AWP; ep6 diagnostic: NOT undertrained.**
On-disk tag is `e39` (study `e39_qwen3_awp`, `output/optuna/e39`) — chosen before the number was
settled; the experiment is **E41** (E39 = word-aug, E40 = qwen3 compression were already taken).
Files keep the `e39` tag (renaming = restarting the live search), mirroring how E38 keeps its `e35`
on-disk tag. Reuses the E28/E38 Optuna harness (`src/optuna_search.py`, `--model` + `--narrow`).

## Motivation

E38's optuna-tuned granite AWP single hit **LB 0.79300** (clears #12) — but every ensemble we have
is **all-granite** (E26/E28/E38), so members share error structure: the E38 AWP pair added only
**+0.0017** on the honest fold. A strong **qwen3-0.6B** single is a *different backbone* → genuinely
decorrelated errors → the ensemble lever with real headroom. Baseline qwen3_ls = LB 0.77921; an
AWP-optuna qwen3 should be a much stronger, decorrelated member for a **granite-AWP + qwen3** combo.

## Design decisions

- **Backbone:** `Qwen/Qwen3-Embedding-0.6B` (595.8M params; qwen3-4B OOMs, 0.6B is the one).
- **1 trial / GPU** (not 2). A single qwen3+AWP run is ~13–17 GB (model+AdamW ~10, AWP weight-backup
  +1–2, activations bs16×512 ~3–5); two on a 32 GB 5090 ≈ 26–34 GB → OOM risk. Measured: **14.5 GB/GPU
  at 1/trial** — fits with headroom, faster per-trial (no contention). Parallelism comes from GPU
  count (8), not packing.
- **Narrow, prior-informed space.** Transfer E38's granite findings as priors so a small budget
  converges: **fix** `eff_batch=16` (dominated E38), `warmup=0.1`, `wd=0.01`, `awp_start=1`; **search
  only** the 4 dims that matter, in narrow bands around the E38 optima. 16 trials (2 waves) suffice.
- **Thread caps** (`OMP/MKL/OPENBLAS/NUMEXPR=12`): 8 concurrent trials on the 256-vCPU box would
  otherwise starve the GPUs on the dataloader/BLAS (E38 lesson). Measured load 9/256, util 91–99%.

## Search space (narrow — `--narrow --search_awp`)

| param | range | note |
|---|---|---|
| `lr` | 8e-6 – 4e-5 (log) | qwen3 regime; E38 granite top was 2.3–4.3e-5 |
| `label_smoothing` | 0.05 – 0.18 | |
| `awp_gamma` | 1e-3 – 4e-3 (log) | around E38 optima (stronger than blind 1e-3) |
| `awp_lr` | 1e-4 – 2.8e-4 (log) | ≈2× the E34 blind default |
| **fixed** | eff_batch=16 · warmup=0.1 · wd=0.01 · awp_start=1 · **epochs=4** · richargs · LS · bf16 | |

Anchor (trial 0, enqueued on one worker): qwen3_ls + AWP defaults — `lr 2e-5 · ls 0.10 · awp_gamma
1e-3 · awp_lr 1e-4` — so every result reads as a Δ vs a known-recipe starting point.

## Objective / protocol

- **`--session_fold 0`** (StratifiedGroupKFold, 56k train / 14k holdout, leak-free) — the SAME split
  as the granite AWP fold models, so a **cross-backbone screen (granite-AWP fold pool + qwen3 fold
  configs) on the 14k is honest**. This is the point: verify the granite+qwen3 ensemble gain *before*
  spending an LB submission.
- Objective = best-epoch macro-F1 on the 14k holdout; epochs fixed at 4 (E38: best-epoch=4; noisy
  slice can't reliably pick an epoch). NopPruner (AWP benefit is late-epoch), TPE n_startup=12.
- Losers auto-deleted, running-best weights kept (screen the survivors).

## Results

14/16 trials completed (t009/t010 killed mid-run to free GPUs for the ep6 diagnostic below).
Per-trial ~2.5–3 h. **Best t011 = 0.7776**; the anchor (t000, qwen3_ls + blind AWP defaults) already
scored 0.7750, so 16 trials of tuning bought only **+0.0026** — the qwen3 surface is FLAT, same
lesson as E38's granite plateau but at a lower level.

| tag | fold-0 14k F1 | lr | ls | awp_gamma | awp_lr |
|---|---|---|---|---|---|
| **t011** | **0.7776** | 1.17e-5 | 0.077 | 2.33e-3 | 1.15e-4 |
| t003 | 0.7755 | 1.21e-5 | 0.102 | 2.46e-3 | 1.41e-4 |
| t002 | 0.7754 | 1.38e-5 | 0.161 | 1.48e-3 | 1.30e-4 |
| t000 ⚓ | 0.7750 | 2.00e-5 | 0.100 | 1.00e-3 | 1.00e-4 |
| t012 | 0.7740 | 2.30e-5 | 0.061 | 2.30e-3 | 1.91e-4 |
| t004 | 0.7739 | 1.14e-5 | 0.174 | 1.60e-3 | 1.65e-4 |
| t014 | 0.7735 | 2.52e-5 | 0.099 | 2.14e-3 | 1.75e-4 |
| t006 | 0.7731 | 8.13e-6 | 0.175 | 1.95e-3 | 1.19e-4 |
| t013 | 0.7730 | 8.97e-6 | 0.126 | 2.31e-3 | 1.02e-4 |
| t015 | 0.7722 | 1.28e-5 | 0.080 | 2.22e-3 | 2.54e-4 |
| t007 | 0.7720 | 2.69e-5 | 0.113 | 3.72e-3 | 1.30e-4 |
| t001 | 0.7716 | 9.21e-6 | 0.155 | 1.98e-3 | 1.34e-4 |
| t005 | 0.7713 | 1.82e-5 | 0.127 | 2.48e-3 | 1.97e-4 |
| t008 | 0.7682 | 3.02e-5 | 0.158 | 2.96e-3 | 1.33e-4 |

Reading: top trials cluster at **lr ~1.2e-5** (lower than granite's 2.3–4.3e-5 optimum) with
`awp_gamma ~2.3e-3`; label smoothing is loose (0.077–0.161 all in the top 3). Spread across 14
trials is only 0.0094 top-to-bottom.

**Verdict: qwen3+AWP ceiling ≈ 0.7776 vs granite AWP fold ≈ 0.7806 → qwen3 stays a ~0.003-weaker
single.** Its only remaining value is cross-backbone ensemble diversity (the screen below decides).

### LB verdict — qwen3 single LOSES to granite on the real leaderboard (2026-07-14)

t011 promoted `--full_data` (ep4, slice 0.7827), vocab-pruned (richargs, 29.7k vocab; parity
800/800 = 100% pruned≡full), shipped as `submit_0714_qwen3_t011.zip`:

| stage | qwen3 t011 | granite AWP t031 | gap |
|---|---|---|---|
| fold-0 14k | 0.7776 | 0.7806 | −0.0030 |
| 3.5k slice | 0.7827 | 0.7859 | −0.0032 |
| **LB** | **0.79130** | **0.79300** | **−0.00170** |

**The E8 "qwen3 loses slice / wins LB" shift PARTIALLY held but was far too small.** qwen3 did climb
relative to granite going CV→LB (gap narrowed from −0.0030/−0.0032 to −0.0017, i.e. ~+0.0013–0.0015
of relative lift), but nothing like the ~+0.016 slice→LB swing E8 showed for the *untuned* LS pair.
Net: **qwen3-0.6B is a genuinely weaker single than granite-311m AWP on this task, confirmed on the
LB, not just CV.** Inference **9:14** (T4) — a qwen3-containing *ensemble* would bust the 10-min cliff,
so its residual diversity value is largely unshippable as a multi-model entry anyway. **Cross-backbone
single/ensemble track CLOSED for the deadline; granite AWP stays the champion.** (SUBMISSIONS row 25.)

### ep6 undertraining diagnostic — NOT undertrained

Hypothesis to kill: qwen3's flat surface = undertraining at 4 epochs (it memorizes train by ep1 but
maybe val hadn't converged). Reran the anchor (t000) and a top config (t002) at **epochs=6** (same
fold, 6-epoch LR schedule), GPUs 6–7:

| epoch | t000 (lr 2e-5) | t002 (lr 1.38e-5) |
|---|---|---|
| 1 | 0.6786 | 0.6877 |
| 2 | 0.7522 | 0.7616 |
| 3 | 0.7670 | 0.7676 |
| **4** | **0.7768** | **0.7757** |
| 5 | 0.7731 ↓ | 0.7749 ↓ |
| 6 | 0.7666 ↓ | 0.7728 ↓ |

Both **peak at epoch 4 and decline monotonically after** — even with the longer schedule's slower LR
decay. The ep4 peaks (0.7768/0.7757) match the 4-epoch-schedule results (~0.775), i.e. more epochs
buy nothing. **qwen3 is converged, not undertrained; epochs=4 confirmed.** The ~0.778 ceiling is
model-level, not schedule-level.

## After the search

1. **Cross-backbone screen** on the 14k: granite-AWP fold pool (t001/t002/t008/t010/t031/t040/t070,
   `output/e38/e35_search/`) + qwen3 fold configs → best qwen3 single + best **granite+qwen3** combo.
2. **Promote** the winner(s) on `--full_data` (epochs 4, best-epoch), then **vocab-prune** (qwen3's
   151k vocab → prune like the qwen3_ls 830M submission; full-vocab qwen3 won't fit the 1 GB cap).
3. ⚠️ **qwen3 runtime:** qwen3_ls ran **9:18** on DACON's T4 (tight under the 10-min cliff). A qwen3
   *single* is fine; a qwen3-containing *ensemble* is cap-risky (slow member) → timing-check any zip.
