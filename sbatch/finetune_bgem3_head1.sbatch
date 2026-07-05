#!/bin/bash
#SBATCH --job-name=bgem3-head1
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3

# TRUE single-linear head (dropout -> Linear 1024->14, no hidden layer, no
# nonlinearity) — the qwen3-style head, completing the depth axis 1/2/3:
#   head1 = THIS | tanh-2 (stock) 0.7568 | gelu-2 0.7515 | gelu-3 0.7317 | tanh-3 queued
# Same recipe as hist0 / the head array; only --head_layers 1 differs.
#   Results: output/pat/ft_results_head1.csv
#   Weights: output/pat/ft_BAAI__bge-m3_head1/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --head_layers 1 --tag head1 \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_head1.csv"
