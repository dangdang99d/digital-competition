#!/bin/bash
#SBATCH --job-name=bgem3-ultralean
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3

# Maximum compression: combo + bare action lines ("grep_search fail" — no
# ACTION marker, no ->; role is unambiguous because action names are atomic
# tokens). Tests how far serialization compression goes before F1 breaks.
# Recipe otherwise IDENTICAL to hist0 (@1024, lr 2e-5, eff. batch 16, 3 ep).
#   Results: output/pat/ft_results_serial_ultralean.csv
#   Weights: output/pat/ft_BAAI__bge-m3_ser_ultralean/  (ship run-dir tokenizer!)

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize ultralean --special_tokens --tag ser_ultralean \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_serial_ultralean.csv"
