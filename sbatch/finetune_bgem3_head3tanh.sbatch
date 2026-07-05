#!/bin/bash
#SBATCH --job-name=bgem3-head3tanh
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3

# Completes the head 2x2 (depth x activation), isolating depth from activation:
#   tanh-2 (stock) 0.7568 | gelu-2 0.7515 | gelu-3 0.7317 | tanh-3 = THIS RUN
# Same recipe as hist0 / the head array; only --head_layers 3 --head_act tanh.
#   Results: output/pat/ft_results_head3tanh.csv
#   Weights: output/pat/ft_BAAI__bge-m3_head3tanh/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --head_layers 3 --head_act tanh --tag head3tanh \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_head3tanh.csv"
