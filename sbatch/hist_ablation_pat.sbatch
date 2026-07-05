#!/bin/bash
#SBATCH --job-name=hist-abl-pat
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3

# History ablation, RESCUED from the dead moana queue (power outage killed task 0
# at 57%, tasks 1-3 never started). Content question #1: does the model need the
# FULL history, or do the last N events suffice (does trimming noise even help)?
# One factor varied, everything else fixed, FROM SCRATCH at max_len 1024 so length
# truncation is never a confound. Task 0 (full) is the from-scratch 1024 baseline.
# On pat's 24GB cards AdamW fits bge-m3 (moana needed adafactor) — all 4 tasks
# share the same optimizer, so they stay internally comparable.
#   Results: output/pat/ft_results_hist_pat_<task>.csv
#   Weights: output/pat/ft_BAAI__bge-m3_hist*/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

HISTS=(0 12 8 4)          # 0 = full history (no cap)
H="${HISTS[$SLURM_ARRAY_TASK_ID]}"

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): max_hist=${H}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --max_hist "${H}" --tag "hist${H}" \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_hist_pat_${SLURM_ARRAY_TASK_ID}.csv"
