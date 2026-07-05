#!/bin/bash
#SBATCH --job-name=bgem3-width
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%A_%a.out

# TARGET MODEL: BAAI/bge-m3

# Width pruning (drop nodes: FFN neurons 4096->2048 + attention heads 16->8,
# importance-selected) — pruning axis #1, vs prune12's depth axis #2.
# Same recipe as prune12: surgery on TRAINED weights, then recover-FT @1024.
#   task 0  width50: hist0 (24L, 0.7568 cal) -> width-pruned    (~50% layer FLOPs,
#           directly comparable to prune12's 0.7619 at the same FLOP cut)
#   task 1  p12w50:  prune12 (12L, 0.7619 cal) -> width-pruned  (#1+#2 combined,
#           ~75% FLOP cut; est. package ~280MB, T4 ~1:30)
#   Results: output/pat/ft_results_width_<task>.csv
#   Weights: output/pat/ft_BAAI__bge-m3_{width50,p12w50}/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

INITS=(output/pat/ft_BAAI__bge-m3_hist0/checkpoint-10500
       output/pat/ft_BAAI__bge-m3_prune12/checkpoint-10500)
TAGS=(width50 p12w50)
INIT="${INITS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "${INIT}/model.safetensors" ]]; then
  echo "ERROR: init checkpoint missing on pat: ${INIT}"
  exit 1
fi

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname): init=${INIT} tag=${TAG}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --init_from "${INIT}" \
  --ffn_keep 0.5 --heads_keep 0.5 --tag "${TAG}" \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_width_${SLURM_ARRAY_TASK_ID}.csv"
