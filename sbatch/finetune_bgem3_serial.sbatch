#!/bin/bash
#SBATCH --job-name=bgem3-serial
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3

# Input-serialization ablation vs hist0 (v1 format, 0.7568 cal). Each variant
# flips exactly ONE axis of the serialized string; recipe otherwise IDENTICAL
# to hist0 (@1024, lr 2e-5, eff. batch 16, 3 epochs, seed 42).
#   task 0  nometa:    no [tier=... ci=... open=...] header line
#   task 1  leanact:   ACTION name -> ok|fail (args/summaries dropped, verdict
#                      rule v2; markers kept — bare lines are combo-only)
#   task 2  dupprompt: prompt text doubled inside the PROMPT line
#   task 3  spectok:   bare action lines "grep_search fail" + 17 atomic tokens
#                      (14 action names + ACTION/USER:/PROMPT: markers, mean-of-
#                      pieces init) — the fully-compressed action representation;
#                      ship the RUN-DIR tokenizer at inference
#   Results: output/pat/ft_results_serial_<task>.csv
#   Weights: output/pat/ft_BAAI__bge-m3_ser_{nometa,leanact,dupprompt,spectok}/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

VARIANTS=(nometa leanact dupprompt bareact)
TAGS=(nometa leanact dupprompt spectok)
EXTRA=("" "" "" "--special_tokens")
VAR="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): serialize=${VAR} tag=ser_${TAG}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

# shellcheck disable=SC2086  # EXTRA is intentionally word-split
"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize "${VAR}" --tag "ser_${TAG}" ${EXTRA[$SLURM_ARRAY_TASK_ID]} \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_serial_${SLURM_ARRAY_TASK_ID}.csv"
