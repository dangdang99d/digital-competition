# Runbook — multi-worker Optuna on a rented vast.ai box

Ops reference (NOT a method report — the science lives in `results_e38.md`). Distilled from
the E38 AWP-optuna launch (2026-07-13, 8×5090, 16 workers; its live run keeps `e35_*`
artifact names — renumbered after launch). Read this before the next multi-worker vast
search; each item is a symptom → cause → fix that cost real time.

## The known-good launch recipe (copy this)

```bash
# per-box, one worker per (gpu, slot). Cap threads, use the venv python, detach.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=2
PY=/venv/main/bin/python                 # torch is in the vast venv, NOT system python3
# launch each worker staggered ~12s (avoid a simultaneous startup spike), e.g.:
CUDA_VISIBLE_DEVICES=$g TQDM_DISABLE=1 nohup $PY -u -m src.optuna_search \
  --gpu_tag g${g}a --n_trials 200 --exp e35 --search_awp \
  --study e35_awp_granite --db output/optuna/e35.db --out_dir output/optuna/e35 \
  > output/optuna/e35/e35_g${g}a.log 2>&1 &
# run the WHOLE launch script detached so it survives the ssh session:
#   setsid nohup bash e35_launch.sh > e35_launch.out 2>&1 < /dev/null &
```
Single box → **plain SQLite** (`--db`). Only use `--journal` (JournalStorage) when 2+ boxes
share NFS (SQLite over NFS corrupts on cross-host locks).

## Gotchas (symptom → cause → fix)

1. **GPUs "running" but only ~40% fed** (util 8–84%, power ~40%, load avg ~340 on 256 cores,
   ~9k python threads). → Each torch proc defaults OMP/BLAS threads to *all* cores, so
   16 workers × 256 = massive oversubscription starves the GPU feed. → `export
   OMP_NUM_THREADS=8` (+ MKL/OPENBLAS/NUMEXPR). Load 340→33, GPU util 8–84%→92–98%. **Biggest
   throughput lever — do this from the start.**

2. **`ModuleNotFoundError: No module named 'optuna'`** — all workers crash instantly. →
   optuna isn't in `requirements.txt` (submission-only deps). → `/venv/main/bin/pip install
   optuna` after provisioning.

3. **Wrong interpreter** — system `python3` has no torch. → torch is in the vast venv. →
   use `/venv/main/bin/python` explicitly (don't rely on `command -v python3`).

4. **`pkill -f optuna_search` silently kills its own shell** (the shell's command line
   contains the pattern) → the rest of the command never runs. → bracket trick:
   `pkill -9 -f "[s]rc.optuna_search"` (regex `[s]rc` matches "src" but the literal pattern
   string doesn't match itself).

5. **Partial launch** — a foreground staggered launch over ssh gets SIGKILL'd (137) mid-ramp
   when the ssh session tears down; only some workers come up (their nohup children survive,
   so you get a confusing partial fleet). → run the launcher itself detached:
   `setsid nohup bash launch.sh &`.

6. **Instance discarded on stop** — stopping a vast instance that's still in `created`/init
   (before it finishes provisioning) *deletes* it, not parks it. → let it reach `running`
   first, or just destroy+re-rent.

7. **Dud host stuck on image pull** (status `loading` for 30+ min, `status_msg` frozen on
   one Docker layer, `direct_port_start: -1`). → slow host internet (the dud had the slowest
   `inet_down`, 870 Mbps). → **rank offers by `inet_down`, not just reliability**; a
   fast-internet host (1700+ Mbps) pulls the multi-GB torch image in ~4 min. Destroy + re-rent
   rather than waiting. Add a watchdog that flags a frozen `status_msg` in minutes.

8. **SSH host-key conflict** — vast reuses `sshN.vast.ai:PORT` endpoints across instances, so
   `known_hosts` has a stale key and ssh is *refused* (looks like a provisioning failure). →
   `ssh-keygen -R "[sshN.vast.ai]:PORT"` and use `-o StrictHostKeyChecking=no -o
   UserKnownHostsFile=/dev/null` for the ephemeral box.

## Pre-launch gate checklist (5090 / Blackwell especially)

- [ ] READY sentinel present, repo at the expected commit, data files present.
- [ ] `pip install optuna` into the venv.
- [ ] **CUDA works on the GPU arch** — the pinned image is built for older cards; on a 5090
  confirm a real op runs: `python -c "import torch; a=torch.randn(2048,2048,device='cuda',
  dtype=torch.bfloat16); (a@a).sum()"` → must not raise "no kernel image for sm_120".
  (torch 2.7.1+cu128 DOES support Blackwell sm_120 — verified.)
- [ ] **VRAM probe** one real trial (bs/max_len of the search) → peak mem decides packing.
  granite-311m AWP @ bs16/len512 = ~13.3 GB → 2/GPU fits a 31 GB 5090 (~26 GB, ~4 GB margin).
- [ ] Launch detached with thread caps; verify load drops and GPU util >90% before walking away.
