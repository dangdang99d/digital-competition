#!/bin/bash
#SBATCH --job-name=ceil-firststep
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# CEILING probe for the FIRST-STEP (zero-history) failure mode, full-FT stock
# BAAI/bge-m3, 14-class head. Both eval on the SAME 1807 real zero-history val
# (seed 42), so directly comparable to each other and to the generalist:
#   generalist hist0 on first-step   0.5550 macroF1 / 0.633 acc
#   generalist qwen3 on first-step   0.5729 macroF1 / 0.629 acc
#
#   task 0  zero_history: train on the 7193 REAL first-step samples (clean labels).
#           Is first-step under-fit (dilution) or an intrinsic no-history ceiling?
#   task 1  strip_history: train on ALL 56k samples with history STRIPPED (10x
#           data, but labels were chosen given history; turn= field is kept, so
#           phase is still visible). Does augmentation beat the clean subset?
# Read-out: task1 >= task0 -> history mostly redundant, augment; task1 < task0 ->
# labels are history-dependent, the clean 7193 win.
#
# AdamW (better than adafactor); full-FT states need a 24GB GPU -> run on a 3090
# (pat/mat/ocean), NOT the 2080 Ti.

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

FLAGS=(--zero_history --strip_history)
TAGS=(ceil_firststep ceil_striphist)
FLAG="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${TAG} (${FLAG})"
nvidia-smi -L || true

source /data/ksjang0515/conda/etc/profile.d/conda.sh
conda activate dacon
export HF_HOME=/data/ksjang0515/hf_cache

python -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize v1 \
  "${FLAG}" \
  --tag "${TAG}" \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --optim adamw_torch \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_firststep_${SLURM_ARRAY_TASK_ID}.csv"
