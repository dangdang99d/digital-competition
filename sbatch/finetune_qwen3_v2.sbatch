#!/bin/bash
#SBATCH --job-name=qwen3-v2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodelist=pat-g2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: Qwen/Qwen3-Embedding-0.6B

# Qwen3-Embedding-0.6B v2 — continue from the 0.7680/0.7752 champion (bf16 run,
# checkpoint-10500), whose epoch curve was still climbing +3pts/epoch at ep3:
#   1. 3 more epochs at lr 1e-5 (undertrained at 3)
#   2. --full_data: +10.5k train samples; eval/calibrate on untouched 3.5k
#   3. max_len 1024: kills the remaining 0.9% truncation (Qwen3 tokenizer)
# bf16 auto-detected on the 4090 (fp16 NaN'd this model; never fp16-train it).
#   Submit:  sbatch sbatch/finetune_qwen3_v2.sbatch
#   Results: output/pat/ft_results_qwen3_v2.csv
#   Weights: output/pat/ft_Qwen__Qwen3-Embedding-0.6B_v2_1024_full/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model Qwen/Qwen3-Embedding-0.6B \
  --init_from ./output/pat/ft_Qwen__Qwen3-Embedding-0.6B/checkpoint-10500 \
  --full_data --tag v2_1024_full \
  --max_len 1024 --epochs 3 --lr 1e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_qwen3_v2.csv"
