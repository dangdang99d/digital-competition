#!/bin/bash
#SBATCH --job-name=bgem3-loraspec
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3 (hist0 checkpoint, frozen base)

# Hierarchical specialists: 4 per-group LoRA adapters (r=16, q/v + head) on the
# SHARED hist0 backbone. Router = multilingual-e5-small group classifier
# (trained locally; bar = hist0's 97.86% group acc). Storage: 1 backbone
# (654MB pkg) + 4 adapters (~11MB each) + router (~86MB pruned) fits the 1GB zip.
#   task 0 explore   task 1 edit   task 2 execute   task 3 noncode
# Specialists see only their group's samples (split-safe --pair filtering);
# eval = within-group macro-F1 on the group's val slice + per-run logit_bias.
# Composition into final 14-class F1 happens locally (compose_hierarchy).
#   Results: output/pat/ft_results_lorah_<task>.csv

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

HIST0=output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500
GROUPS=(explore edit execute noncode)
CLS=("read_file,grep_search,list_directory,glob_pattern"
     "edit_file,write_file,apply_patch"
     "run_bash,run_tests,lint_or_typecheck"
     "ask_user,plan_task,web_search,respond_only")
G="${GROUPS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "${HIST0}/model.safetensors" ]]; then
  echo "ERROR: hist0 checkpoint missing on pat: ${HIST0}"
  exit 1
fi

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): group=${G}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -m pip show peft >/dev/null 2>&1 || "${PY}" -m pip install "peft>=0.15"

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize v1 \
  --init_from "${HIST0}" \
  --pair "${CLS[$SLURM_ARRAY_TASK_ID]}" \
  --lora 16 \
  --tag "lora_${G}" \
  --max_len 1024 --epochs 2 --lr 1e-4 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_lorah_${SLURM_ARRAY_TASK_ID}.csv"
