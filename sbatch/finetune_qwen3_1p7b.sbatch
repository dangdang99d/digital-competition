#!/bin/bash
#SBATCH --job-name=qwen3-1p7b
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: Qwen/Qwen3-1.7B

# Qwen3-1.7B capacity test — does ~3x params beat the 0.6B champion (0.7680)?
# Not directly submittable (fp16 3.4GB > 1GB zip cap): this is a MEASUREMENT of
# capacity headroom and, if strong, the distillation TEACHER (big model -> small
# student). Adafactor: AdamW state doesn't fit 1.7B on 24GB; Adafactor cost ~0
# (bge-m3 champion was Adafactor). bf16 auto on 3090/4090 (never fp16 for Qwen3).
# Any batch-partition GPU works (all 3090/4090) — no nodelist pin.
#   Results: output/pat/ft_results_qwen3_1p7b.csv

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model Qwen/Qwen3-1.7B \
  --optim adafactor --lr 2e-5 \
  --max_len 512 --epochs 3 \
  --batch_size 2 --grad_accum 8 \
  --out_dir ./output/pat \
  --results_name "ft_results_qwen3_1p7b.csv"
