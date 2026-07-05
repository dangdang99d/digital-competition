#!/bin/bash
#SBATCH --job-name=bgem3-prune
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3

# Depth pruning 24 -> 12 layers (evenly spaced, first+last kept), initialized from
# the trained champion, then fine-tuned to recover. Halves inference FLOPs — the
# insurance policy for the 10-min T4 limit at max_len 1024 — and tells us how much
# F1 half the depth costs. The pruned config reloads cleanly with from_pretrained
# (num_hidden_layers is rewritten), so prune_vocab.py + packaging work unchanged.
#
# REQUIRES the champion checkpoint on pat first (it lives on moana/local):
#   rsync -avz output/ft_BAAI__bge-m3/checkpoint-10500 \
#         ksjang0515@pat:/NAS2/ksjang0515/dacon/output/ft_BAAI__bge-m3/
#   Results: output/pat/ft_results_bgem3_prune.csv
#   Weights: output/pat/ft_BAAI__bge-m3_prune12/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

CKPT=output/ft_BAAI__bge-m3/checkpoint-10500
if [[ ! -f "${CKPT}/model.safetensors" ]]; then
  echo "ERROR: champion checkpoint missing on pat — upload it first:"
  echo "  rsync -avz output/ft_BAAI__bge-m3/checkpoint-10500 ksjang0515@pat:/NAS2/ksjang0515/dacon/output/ft_BAAI__bge-m3/"
  exit 1
fi

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --init_from "${CKPT}" --keep_layers 12 --tag prune12 \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_bgem3_prune.csv"
