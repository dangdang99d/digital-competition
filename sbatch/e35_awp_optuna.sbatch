#!/bin/bash
#SBATCH --job-name=e35_awp
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out
#
# E35 · AWP hyperparameter search (Optuna, joint 9-param) — SLURM version.
# Runs ONE Optuna worker per job on ONE GPU (--gres=gpu:1). Submit this TWICE
# (worker 0 and worker 1) → the two jobs drive ONE shared distributed study, so
# together they explore 2 x N_TRIALS trials in parallel, one per GPU.
#
# Storage = JournalStorage on shared NFS (--journal), NOT SQLite: the two jobs may
# land on DIFFERENT nodes and SQLite over NFS corrupts under cross-host locking; the
# journal's symlink lock is NFS-safe. Both workers point at the SAME journal path.
#
# Usage (submit worker 0 FIRST so it creates the study + enqueues the anchor, then
# worker 1 a few seconds later):
#   sbatch sbatch/e35_awp_optuna.sbatch 0     # worker 0 -> enqueues anchor trial 0
#   sbatch sbatch/e35_awp_optuna.sbatch 1     # worker 1 -> joins the same study
#
# Check progress:  see the "monitor" block printed at the end of this script.
set -euo pipefail

WORKER="${1:?usage: sbatch $0 <worker 0 or 1>}"

# --- project root: the dir you submitted from (put the repo there) ---
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- python: the dacon env by ABSOLUTE PATH (no conda activate in batch shells) ---
# Edit if your env lives elsewhere:  conda env list  -> <path>/bin/python
PY="${DACON_PY:-$HOME/miniconda3/envs/dacon/bin/python}"

echo "job ${SLURM_JOB_ID} worker=${WORKER} on $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

# --- shared study config (identical on BOTH workers) ---
STUDY="e35_awp_granite"
JOURNAL="output/optuna/e35_journal.log"   # shared NFS path -> both workers = one study
OUT_DIR="output/optuna/e35"
N_TRIALS=30                                # 2 workers x 30 = 60 trials total
mkdir -p "${OUT_DIR}/logs" sbatch/logs

# worker 0 seeds the LB-validated AWP config as anchor trial 0; worker 1 does not.
ANCHOR=""
if [[ "${WORKER}" == "0" ]]; then ANCHOR="--enqueue_anchor"; fi

# --gpu_tag MUST be unique per worker (per-worker results CSV: no append race).
"${PY}" -u -m src.optuna_search \
  --gpu_tag "w${WORKER}" \
  --exp e35 --search_awp ${ANCHOR} \
  --study "${STUDY}" \
  --journal "${JOURNAL}" \
  --out_dir "${OUT_DIR}" \
  --n_trials "${N_TRIALS}" \
  --fold 0

echo "----"
echo "worker w${WORKER} finished. results:"
echo "  study journal : ${JOURNAL}"
echo "  trial logs    : ${OUT_DIR}/logs/trial_*.log"
echo "  per-worker CSV: output/pat/optuna_w${WORKER}.csv (or --out_dir results)"
echo "monitor best-so-far across BOTH workers:"
echo "  ${PY} -c \"import optuna;from optuna.storages import JournalStorage;from optuna.storages.journal import JournalFileBackend,JournalFileSymlinkLock as L;s=JournalStorage(JournalFileBackend('${JOURNAL}',lock_obj=L('${JOURNAL}')));st=optuna.load_study(study_name='${STUDY}',storage=s);print('trials',len(st.trials));print('best',st.best_value,st.best_params)\""
