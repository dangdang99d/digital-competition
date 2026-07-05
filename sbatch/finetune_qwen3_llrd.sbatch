#!/bin/bash
#SBATCH --job-name=qwen3-llrd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: Qwen/Qwen3-Embedding-0.6B

# LLRD A/B on the champion recipe: IDENTICAL to the 0.7680 run (Qwen3-0.6B,
# AdamW 2e-5, bf16, 3ep, max_len 512, from scratch) with ONE change: --llrd 0.9
# (layer i gets lr*0.9^(depth-i); embeddings lowest, head full lr).
# Single-factor comparison: beat 0.7680 -> adopt LLRD for the Qwen3 line.
#   Results: output/pat/ft_results_qwen3_llrd.csv

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model Qwen/Qwen3-Embedding-0.6B \
  --llrd 0.9 --tag llrd09 \
  --max_len 512 --epochs 3 --lr 2e-5 \
  --batch_size 8 --grad_accum 2 \
  --out_dir ./output/pat \
  --results_name "ft_results_qwen3_llrd.csv"
