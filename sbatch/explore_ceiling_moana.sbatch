#!/bin/bash
#SBATCH --job-name=ceil-explore
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# CEILING probe for the router-adapter (triage) approach: full fine-tune of
# STOCK BAAI/bge-m3 (24L, fresh from HF — no prune, no frozen base, no LoRA)
# on ONLY the explore-group samples (~23k train / 5.7k val), standard recipe.
# This is the most capacity any explore specialist could ever get. Read-out
# against the same 4-class explore-restricted macro-F1:
#   flat bge@512 restricted      0.5988 cal
#   LoRA r16 on prune12 (2ep)    0.6011 cal
#   THIS RUN = the ceiling. If it also lands ~0.60, the explore group is
#   noise-bounded and no specialist architecture can improve it; if it clears
#   ~0.62+, specialist capacity matters and the adapter design needs revisiting.

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname): explore ceiling probe"
nvidia-smi -L || true

source /data/ksjang0515/conda/etc/profile.d/conda.sh
conda activate dacon
export HF_HOME=/data/ksjang0515/hf_cache

python -u -m src.finetune \
  --model BAAI/bge-m3 \
  --serialize v1 \
  --pair "read_file,grep_search,list_directory,glob_pattern" \
  --tag ceil_explore \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_ceil_explore.csv"
