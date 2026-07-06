#!/bin/bash
#SBATCH --job-name=bgem3-rich
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3

# names-inspired metadata ladder (teammate's granite result attributes the gain
# to dropping directory paths). One axis per rung, all vs hist0 (0.7568 cal):
#   task 0  richmeta: basename open files + count, bucketed budget/elapsed,
#                     codelang, loc. Structure/history unchanged.
#   task 1  richargs: richmeta + directory simplification inside history action
#                     args (path/scope/target/glob values). result_summary still
#                     carries full paths (documented boundary; richall = later rung).
#   Read-out: richmeta - hist0 = metadata axis; richargs - richmeta = args axis.
#   Results: output/pat/ft_results_rich_<task>.csv

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

VARIANTS=(richmeta richargs)
VAR="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): serialize=${VAR}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize "${VAR}" \
  --tag "ser_${VAR}" \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_rich_${SLURM_ARRAY_TASK_ID}.csv"
