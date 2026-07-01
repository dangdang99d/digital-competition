#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# LoRA fine-tuning (single linear head) for the 14-class next-action task.
# Runs the 3-model bake-off sequentially: frozen-probe winner, the LB-68.8 baseline,
# and the efficient runner-up. Each ~1-2h; all three fit the 1-day wall.
# Moana cluster. No --account/--qos needed.
#   Submit:  sbatch sbatch/finetune.sbatch
#   Results: output/ft_results.csv  (+ output/ft_<model>/ checkpoints)

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python
export HF_HOME=/data/ksjang0515/hf_cache

for MODEL in \
  Qwen/Qwen3-Embedding-0.6B \
  ibm-granite/granite-embedding-311m-multilingual-r2 \
  Alibaba-NLP/gte-multilingual-base ; do
  echo "=== fine-tuning ${MODEL} ==="
  "${PY}" -u -m src.finetune --model "${MODEL}" --epochs 3 --out_dir ./output
done
