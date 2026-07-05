#!/bin/bash
#SBATCH --job-name=bgem3-v2
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3

# bge-m3 v2: CONTINUE training from the 0.7349 checkpoint with two upgrades:
#   1. max_len 512 -> 1024: no sample is truncated anymore (longest = 650 tokens;
#      4.1% used to lose their PROMPT: line — the message being classified).
#   2. --full_data: 75% of the old val folds into training (+10.5k samples);
#      eval + calibration use the untouched 25% (3.5k) the checkpoint never saw.
# Continued FT: lower lr (1e-5), 2 epochs, adafactor (fits 2080 Ti at 568M).
# --tag v2_1024_full keeps the run dir separate from the champion checkpoint.
#   Submit:  sbatch sbatch/finetune_bgem3_v2.sbatch
#   Results: output/ft_results_bgem3_v2.csv
#   Weights: output/ft_BAAI__bge-m3_v2_1024_full/

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python
export HF_HOME=/data/ksjang0515/hf_cache

# batch 2 x accum 8 = same effective batch 16; halved batch for the 2x seq len.
"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --init_from ./output/ft_BAAI__bge-m3/checkpoint-10500 \
  --full_data --tag v2_1024_full \
  --max_len 1024 --epochs 2 --lr 1e-5 --optim adafactor \
  --batch_size 2 --grad_accum 8 \
  --keep_checkpoints 3 \
  --out_dir ./output \
  --results_name "ft_results_bgem3_v2.csv"
