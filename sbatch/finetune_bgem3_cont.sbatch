#!/bin/bash
#SBATCH --job-name=bgem3-cont
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3

# Continuation training: +2 epochs from FINISHED checkpoints at half LR (1e-5).
# Tests whether the 3-epoch schedule left accuracy on the table (all curves
# still climbed into the final LR decay). NOTE: this is a fresh warmup+decay
# cycle (SGDR-style restart), not an extension of the original schedule.
#   task 0  cont_hist0:   v1 baseline 0.7568 cal — the reference for the axis
#   task 1  cont_spectok: bareact+atomic 0.7411 — new tokens may want more steps
#                         (embedding guard keeps their trained rows: fixed in
#                         finetune.py add_action_tokens)
#   task 2  cont_prune12: 12L champion 0.7619 — its curve was still climbing
#   Results: output/pat/ft_results_cont_<task>.csv
#   Weights: output/pat/ft_BAAI__bge-m3_cont_{hist0,spectok,prune12}/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

INITS=(output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500
       output/pat/ft_BAAI__bge-m3_ser_spectok/checkpoint-10500
       output/pat/ft_BAAI__bge-m3_prune12/checkpoint-10500)
TAGS=(cont_hist0 cont_spectok cont_prune12)
SER=(v1 bareact v1)
EXTRA=("" "--special_tokens" "")
INIT="${INITS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "${INIT}/model.safetensors" ]]; then
  echo "ERROR: init checkpoint missing on pat: ${INIT}"
  exit 1
fi

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): init=${INIT} tag=${TAG}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

# shellcheck disable=SC2086  # EXTRA is intentionally word-split
"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --init_from "${INIT}" \
  --serialize "${SER[$SLURM_ARRAY_TASK_ID]}" ${EXTRA[$SLURM_ARRAY_TASK_ID]} \
  --tag "${TAG}" \
  --max_len 1024 --epochs 2 --lr 1e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_cont_${SLURM_ARRAY_TASK_ID}.csv"
