#!/bin/bash
#SBATCH --job-name=bgem3-serial
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-2
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
#   task 1  leanact:   ACTION name -> ok|fail  (args + result_summary dropped)
#   task 2  dupprompt: PROMPT line emitted twice (prompt-emphasis probe)
#   Results: output/pat/ft_results_serial_<task>.csv
#   Weights: output/pat/ft_BAAI__bge-m3_ser_{nometa,leanact,dupprompt}/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

VARIANTS=(nometa leanact dupprompt)
VAR="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): serialize=${VAR}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize "${VAR}" --tag "ser_${VAR}" \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_serial_${SLURM_ARRAY_TASK_ID}.csv"
