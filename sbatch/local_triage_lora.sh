#!/bin/bash
# Local (RTX 4060, 8GB) run of triage-lora tasks 1-3 — the moana array tasks
# stuck PD on QOSMaxGRESPerUser. Identical training params to lora_spec_moana.sbatch
# (task 0 = explore already done on moana -> ft_results_lorah_0.csv, cal 0.6011).
# LoRA r=16 on frozen prune12 (12-layer) backbone -> fits 8GB. One GPU -> serial.
set -euo pipefail
cd /home/kyusang/research/dacon
mkdir -p sbatch/logs output/pat

PY=.venv/bin/python
BASE=output/pat/ft_BAAI__bge-m3_prune12/checkpoint-10500
GRP_NAMES=(explore edit execute noncode)
CLS=("read_file,grep_search,list_directory,glob_pattern"
     "edit_file,write_file,apply_patch"
     "run_bash,run_tests,lint_or_typecheck"
     "ask_user,plan_task,web_search,respond_only")

for i in 1 2 3; do
  G="${GRP_NAMES[$i]}"
  LOG="sbatch/logs/local-triage-lora_${i}.out"
  echo "==== task ${i} group=${G} -> ${LOG} ($(date '+%F %T')) ====" | tee -a "${LOG}"
  "${PY}" -u -m src.finetune \
    --model BAAI/bge-m3 \
    --serialize v1 \
    --init_from "${BASE}" \
    --pair "${CLS[$i]}" \
    --lora 16 \
    --tag "lora_p12_${G}" \
    --max_len 1024 --epochs 2 --lr 1e-4 \
    --batch_size 4 --grad_accum 4 \
    --out_dir ./output/pat \
    --results_name "ft_results_lorah_${i}.csv" >> "${LOG}" 2>&1
  echo "==== task ${i} DONE ($(date '+%F %T')) ====" | tee -a "${LOG}"
done
echo "ALL DONE ($(date '+%F %T'))"
