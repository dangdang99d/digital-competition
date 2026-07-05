#!/bin/bash
#SBATCH --job-name=hist-abl
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3

# History ablation (#6): does the model need the FULL history, or do the last
# N events suffice (and does trimming noise even help)? One factor varied,
# everything else = champion recipe at max_len 1024 (so length truncation is
# never a confound), FROM SCRATCH so runs are directly comparable to each other.
# Task 0 (full) doubles as the from-scratch 1024 baseline vs the continued-v2 run.
# Baseline model = bge-m3 (the champion we ship; granite findings might not transfer).
#   Results: output/ft_results_hist_<n>.csv   Weights: output/ft_BAAI__bge-m3_hist*/

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output

HISTS=(0 12 8 4)          # 0 = full history (no cap)
H="${HISTS[$SLURM_ARRAY_TASK_ID]}"

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): max_hist=${H}"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python
export HF_HOME=/data/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --max_hist "${H}" --tag "hist${H}" \
  --max_len 1024 --epochs 3 --lr 2e-5 --optim adafactor \
  --batch_size 2 --grad_accum 8 \
  --out_dir ./output \
  --results_name "ft_results_hist_${SLURM_ARRAY_TASK_ID}.csv"
