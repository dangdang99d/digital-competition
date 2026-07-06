#!/bin/bash
#SBATCH --job-name=triage-lora
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# MOANA version of finetune_bgem3_lora_spec.sbatch — identical training
# parameters, only paths/partition/env differ (conventions from commands_moana).
# Triage specialists: 4 per-group LoRA adapters (r=16) on the frozen prune12
# backbone; self-routing via the base 14-class head. See the pat file's header.
#   task 0 explore   task 1 edit   task 2 execute   task 3 noncode
# out_dir stays ./output/pat so checkpoint/results layout matches the local &
# pat trees (rsync-back merges cleanly).

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

BASE=output/pat/ft_BAAI__bge-m3_prune12/checkpoint-10500
GRP_NAMES=(explore edit execute noncode)
CLS=("read_file,grep_search,list_directory,glob_pattern"
     "edit_file,write_file,apply_patch"
     "run_bash,run_tests,lint_or_typecheck"
     "ask_user,plan_task,web_search,respond_only")
G="${GRP_NAMES[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "${BASE}/model.safetensors" ]]; then
  echo "ERROR: prune12 checkpoint missing on moana: ${BASE} (rsync it first)"
  exit 1
fi

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): group=${G}"
nvidia-smi -L || true

source /data/ksjang0515/conda/etc/profile.d/conda.sh
conda activate dacon
export HF_HOME=/data/ksjang0515/hf_cache

python -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize v1 \
  --init_from "${BASE}" \
  --pair "${CLS[$SLURM_ARRAY_TASK_ID]}" \
  --lora 16 \
  --tag "lora_p12_${G}" \
  --max_len 1024 --epochs 2 --lr 1e-4 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_lorah_${SLURM_ARRAY_TASK_ID}.csv"
