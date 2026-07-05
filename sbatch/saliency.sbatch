#!/bin/bash
#SBATCH --job-name=saliency
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3

# Saliency diagnostic (#5) on the bge-m3 champion: gradient-x-input attribution
# on wrong vs correct val predictions, aggregated into meta/history/prompt
# shares. Read the aggregate lines at the end of the log first; per-sample
# detail in output/saliency_bgem3.csv (fetch with the results rsync).

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output

echo "job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python
export HF_HOME=/data/ksjang0515/hf_cache

"${PY}" -u -m src.saliency \
  --ckpt output/ft_BAAI__bge-m3/checkpoint-10500 \
  --n_wrong 400 \
  --out output/saliency_bgem3.csv
