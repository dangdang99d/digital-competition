#!/bin/bash
#SBATCH --job-name=fullft
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --array=0-4
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

# array index -> model + optimizer. AdamW full-FT fits the 2080 Ti (10.75GB)
# only up to ~350M params (granite-311m worked; Qwen3-0.6B/596M OOMed). The
# 560M class runs with ADAFACTOR instead: factored optimizer state (~MBs vs
# Adam's 4.8GB) fits easily, and quality is near-AdamW — unlike SGD, which is
# banned for results (−37 F1 pts). pat's 24GB cards can re-confirm with AdamW.
MODELS=(
  "Alibaba-NLP/gte-multilingual-base"      # 305M — AdamW
  "microsoft/mdeberta-v3-base"             # 278M — AdamW
  "BAAI/bge-m3"                            # 568M — adafactor
  "nlpai-lab/KURE-v1"                      # 568M — adafactor
  "FacebookAI/xlm-roberta-large"           # 561M — adafactor
)
OPTIMS=(
  "adamw_torch"
  "adamw_torch"
  "adafactor"
  "adafactor"
  "adafactor"
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
OPTIM="${OPTIMS[$SLURM_ARRAY_TASK_ID]}"

echo "array task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_JOB_ID}) on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
echo "model: ${MODEL}"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python
export HF_HOME=/data/ksjang0515/hf_cache
# non-interactive: auto-approve any trust_remote_code prompt (gte ships custom code).
# Training flags stay identical across models; this only unblocks gte's load.
# lr 2e-5 — the recipe behind granite's 0.7153 (works for both optimizers).
echo "optim: ${OPTIM}"
yes y 2>/dev/null | "${PY}" -u -m src.finetune \
  --model "${MODEL}" --epochs 3 --out_dir ./output \
  --batch_size 4 --grad_accum 4 --optim "${OPTIM}" --lr 2e-5 \
  --results_name "ft_results_adamw_${SLURM_ARRAY_TASK_ID}.csv"
