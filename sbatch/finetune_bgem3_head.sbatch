#!/bin/bash
#SBATCH --job-name=bgem3-head
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3

# Classification-head depth: replace the stock XLM-R head (dense+tanh+out_proj,
# ~2 layers already) with a fresh 2- or 3-layer GELU MLP. Same recipe as the
# Jul 4 ablation batch, so compare directly against hist0 = 0.7498/0.7568 cal
# (full history, stock head, from scratch @1024, AdamW+bf16, eff. batch 16).
#   Results: output/pat/ft_results_head_<task>.csv
#   Weights: output/pat/ft_BAAI__bge-m3_head{2,3}/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

HEADS=(2 3)
H="${HEADS[$SLURM_ARRAY_TASK_ID]}"

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): head_layers=${H}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --head_layers "${H}" --tag "head${H}" \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_head_${SLURM_ARRAY_TASK_ID}.csv"
