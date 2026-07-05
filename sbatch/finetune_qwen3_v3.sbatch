#!/bin/bash
#SBATCH --job-name=qwen3-v3
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: Qwen/Qwen3-Embedding-0.6B

# Qwen3 v3 — continuation OF v2: submit AFTER qwen3-v2 finishes. Picks up v2's
# best checkpoint and trains 3 more epochs at a further-lowered LR (5e-6),
# same data regime as v2 (--full_data, max_len 1024, bf16 auto).
# Rationale: the champion's curve was still +3/epoch at ep3; if v2 (ep4-6) is
# still climbing, v3 (ep7-9) collects the rest. If v2 plateaued, skip v3.
#   Submit:  sbatch sbatch/finetune_qwen3_v3.sbatch
#   Results: output/pat/ft_results_qwen3_v3.csv
#   Weights: output/pat/ft_Qwen__Qwen3-Embedding-0.6B_v3_1024_full/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

# resolve v2's checkpoint (save_total_limit=1 keeps only the best one)
CKPT=$(ls -d output/pat/ft_Qwen__Qwen3-Embedding-0.6B_v2_1024_full/checkpoint-* | sort -V | tail -1)
echo "init_from: ${CKPT}"

"${PY}" -u -m src.finetune \
  --model Qwen/Qwen3-Embedding-0.6B \
  --init_from "${CKPT}" \
  --full_data --tag v3_1024_full \
  --max_len 1024 --epochs 3 --lr 5e-6 \
  --batch_size 4 --grad_accum 4 \
  --keep_checkpoints 3 \
  --out_dir ./output/pat \
  --results_name "ft_results_qwen3_v3.csv"
