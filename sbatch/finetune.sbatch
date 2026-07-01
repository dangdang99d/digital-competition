#!/bin/bash
#SBATCH --job-name=fullft
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# FULL fine-tuning (all backbone weights + single linear head) for the 14-class task.
# Slurm JOB ARRAY: one independent job per model (array tasks 0..2), so each gets its
# own GPU, log, and full time budget. QoS MaxJobsPU=2 -> up to 2 run at once, 3rd queues.
# Moana cluster. No --account/--qos needed.
#   Submit:  sbatch sbatch/finetune.sbatch
#   Logs:    sbatch/logs/fullft-<jobid>_<taskid>.out  (one per model)
#   Results: output/ft_results_<n>.csv per task  (merge after; no concurrent-append race)

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output

# array index -> model
MODELS=(
  "Qwen/Qwen3-Embedding-0.6B"
  "ibm-granite/granite-embedding-311m-multilingual-r2"
  "Alibaba-NLP/gte-multilingual-base"
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "array task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_JOB_ID}) on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
echo "model: ${MODEL}"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python
export HF_HOME=/data/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model "${MODEL}" --epochs 3 --out_dir ./output \
  --batch_size 4 --grad_accum 4 \
  --results_name "ft_results_${SLURM_ARRAY_TASK_ID}.csv"
