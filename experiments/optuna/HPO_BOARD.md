# Hyperparameter-Tuning Program Board — `research/optuna`

Single management doc for **all** hyperparameter search (Optuna/TPE) work. Same convention
as the compression board (E31): one status row per HPO campaign → the detailed design + numbers
live in the linked per-experiment doc. All scores **raw uncalibrated macro-F1** (project
invariant: NO logit calibration). **LB is the final judge** — the local fold/slice mis-ranks
(E8/E26 CV→LB reversal), so the search only *selects candidates*; a promoted config is only
"won" once the eval-server LB confirms it.

## Shared protocol (applies to every HPO campaign here)

- **Harness:** `src/optuna_search.py` — one worker process per GPU, all workers share one SQLite
  TPE study; each trial launches `src.finetune` as a crash-isolated subprocess on the
  **session-grouped fold-0** split (leak-free; the row-stratified / `--full_data` slices mis-rank).
- **Objective:** fold-0 **best-epoch** macro-F1 (finetune.py RAM-snapshot).
- **Pruning:** MedianPruner (E28: n_startup 8 / warmup 1 · E35: n_startup 12 / warmup 2 for the
  bigger space + AWP's late-appearing effect).
- **Anchor trick:** the LB-validated champion/E34 config is `study.enqueue_trial`'d as trial 0
  (one worker only, `--enqueue_anchor`) so every trial reads as a Δ against a point whose real LB
  value we know.
- **Promotion:** top-2–3 retrain from scratch via **`--full_data`** (best-epoch snapshot), **NEVER
  `--all_data`** — the E28 last-epoch trap cratered t043 to LB 0.75934 (−0.018) by shipping the
  epoch-5 overfit tail. Winners keep `--init_seed 42` (soup/ensemble-compatible).
- **Infra:** rented **vast.ai 4× RTX 3090** (see [VAST.md](../../VAST.md)) — **STEP 0 mandatory
  rental-proposal + user approval before any instance is created.** Study DB + trial logs + kept
  checkpoints land on NFS `output/optuna/`.

## Board

| ID | Campaign | Space | Baseline (LB) | Result | Status |
|----|----------|-------|---------------|--------|--------|
| **E28** | Base granite recipe (champion: granite-311m · richargs · LS · bf16) | 6 params: `lr` · `epochs` · `eff_batch` · `warmup_ratio` · `label_smoothing` · `weight_decay` | granite_ls **0.77738** · anchor fold 0.7586 | ✅ **DONE — SOTA.** best t043 fold 0.7724 (+0.0138 vs anchor); `--full_data` promote **t043fd LB 0.78155** = single-model SOTA. lr-spread trio **LB 0.78780 = overall SOTA**. Fold predicted LB; `--all_data` v1 failed (last-epoch tail). | ✅ |
| **E35** | **AWP knobs + base recipe, JOINT** | 9 params: E28's 6 + **`awp_gamma`** (log 5e-4→5e-3) · **`awp_lr`** (log 3e-5→3e-4) · **`awp_start_epoch`** {0,1} | AWP single LB **0.78557** (anchor, trial 0) · E28 t043fd 0.78155 · trio 0.78780 | — | 🔲 **QUEUED — the untried HPO.** Code ready + dry-run verified; awaiting rental GO. |

Detailed designs & full result tables: **E28** → [results.md](results.md) · **E35** →
[results_e35.md](results_e35.md).

## Untried backlog (not yet specced/queued — candidates, lower priority than E35)

- **qwen3 recipe Optuna** — E28 only tuned granite; qwen3_ls is the *other* co-SOTA single model
  (LB 0.77921). qwen3 recipe was never Optuna-tuned. Lower priority: qwen3 lost granite on LB and
  costs ~9:06 server-time (near the 10-min cap → thin ensemble headroom). Not code-blocked
  (`--model` swap + widen ranges), but not yet specced.
- **Ensemble-diversity-objective HPO** — E28/E35 optimize single-model fold F1; the LB wins came
  from *ensembling* lr-spread members. A search whose objective rewards decorrelated members
  (not peak single F1) is conceptually untried. Needs an objective redesign — parked.
- **AWP + other-lever joint (post-E35)** — only meaningful once E35 says whether AWP's optimum
  shifts the base recipe; gated on E35 read-out.

## Untried HPO to run now → **E35 (AWP Optuna)**

Highest-value open campaign: AWP is the current single-model SOTA lever (LB 0.78557, beats the
E28-tuned recipe alone) but ships with **blind, un-tuned knobs** — clear headroom, and the E28
machinery already turned a fold gain into a real LB gain. Code is additive + dry-run verified.

**Launch (shared SLURM server — 2 jobs, 1 GPU each, one shared distributed study).**
Script: [../../sbatch/e35_awp_optuna.sbatch](../../sbatch/e35_awp_optuna.sbatch). Submit from
the repo root (so `SLURM_SUBMIT_DIR` = repo). Worker 0 FIRST (seeds anchor trial 0 = champion
recipe + AWP defaults, LB 0.78557), then worker 1:

```bash
sbatch sbatch/e35_awp_optuna.sbatch 0   # worker 0 -> --enqueue_anchor
sbatch sbatch/e35_awp_optuna.sbatch 1   # worker 1 -> joins same study
```

- **Storage = JournalStorage on shared NFS** (`--journal output/optuna/e35_journal.log`), NOT
  SQLite — the two jobs can land on different nodes and SQLite over NFS corrupts under cross-host
  locking; both workers point at the SAME journal path → one distributed study.
- Each worker `--n_trials 30` → **60 trials total** (2 GPU × 30), TPE + MedianPruner
  (n_startup 12 / warmup 2). Unique `--gpu_tag w0/w1` (per-worker CSV, no append race).
- **Prereq:** `dacon` conda env installed (`conda env create -f environment.yaml`); the script
  calls `$HOME/miniconda3/envs/dacon/bin/python` by absolute path (override via `DACON_PY`).
- **Check performance** (best-so-far across both workers): run the monitor one-liner the script
  echoes at the end, or `tail sbatch/logs/e35_awp-*.out` / `output/optuna/e35/logs/trial_*.log`.

Promote top-2–3 via `--full_data` (best-epoch, NEVER `--all_data`), package, submit, then fill
E35 [results_e35.md](results_e35.md) §Results + this board's E35 row.

## Log

- **2026-07-13** — board created (this file). E28 closed (SOTA); E35 identified as the sole
  code-ready untried HPO. **Launch path = shared SLURM server** (user runs via sbatch, not vast):
  wrote [../../sbatch/e35_awp_optuna.sbatch](../../sbatch/e35_awp_optuna.sbatch) — 2 jobs × 1 GPU,
  shared NFS JournalStorage study, worker 0 seeds the anchor. Ready to `sbatch` once the `dacon`
  env is installed on the server.
