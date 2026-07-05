#!/bin/bash
#SBATCH --job-name=bgem3-reinit
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3

# Re-init top 2 encoder layers before full-FT: bge-m3's retrieval-pretrained top
# layers may transfer worse to classification than a fresh start. From scratch at
# 1024 — compare against hist-abl-pat task 0 (identical recipe, no re-init).
# --keep_checkpoints 3 -> post-hoc SWA averaging via analysis/swa_average.py.
#   Results: output/pat/ft_results_bgem3_reinit.csv
#   Weights: output/pat/ft_BAAI__bge-m3_reinit2/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --reinit_layers 2 --tag reinit2 \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --keep_checkpoints 3 \
  --out_dir ./output/pat \
  --results_name "ft_results_bgem3_reinit.csv"
