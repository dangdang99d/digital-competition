#!/bin/bash
#SBATCH --job-name=gte-v2
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: Alibaba-NLP/gte-multilingual-base

# gte v2 — SUBMISSION-TRACK model (fp16 ~610MB, fits the <1GB zip limit that
# rules out bge-m3/560M-class). Continue from the AdamW 0.7300 checkpoint with:
#   1. max_len 512 -> 1024 (no sample truncated; 4.1% used to lose PROMPT:)
#   2. --full_data (+10.5k train samples; eval/calib on untouched 3.5k)
# 305M fits AdamW on the 2080 Ti even at seq 1024 (batch 2 x accum 8).
#   Submit:  sbatch sbatch/finetune_gte_v2.sbatch
#   Results: output/ft_results_gte_v2.csv
#   Weights: output/ft_Alibaba-NLP__gte-multilingual-base_v2_1024_full/

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python
export HF_HOME=/data/ksjang0515/hf_cache

# resolve the surviving AdamW checkpoint dir (name is checkpoint-<step>)
CKPT=$(ls -d output/ft_Alibaba-NLP__gte-multilingual-base/checkpoint-* | sort -V | tail -1)
echo "init_from: ${CKPT}"

yes y 2>/dev/null | "${PY}" -u -m src.finetune \
  --model Alibaba-NLP/gte-multilingual-base \
  --init_from "${CKPT}" \
  --full_data --tag v2_1024_full \
  --max_len 1024 --epochs 2 --lr 1e-5 \
  --batch_size 2 --grad_accum 8 \
  --keep_checkpoints 3 \
  --out_dir ./output \
  --results_name "ft_results_gte_v2.csv"
