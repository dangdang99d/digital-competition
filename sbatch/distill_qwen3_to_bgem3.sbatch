#!/bin/bash
#SBATCH --job-name=distill-bgem3
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3  (student; teacher = Qwen/Qwen3-Embedding-0.6B champion)

# Distillation: qwen3-0.6B champion (0.7752 val / 0.76698 LB, already on pat)
# teaches bge-m3 from scratch via soft labels. Cross-tokenizer is fine — only the
# 14-class logits transfer. Single job, two stages:
#   1. dump teacher logits over all 70k train samples (skipped if already dumped)
#   2. train bge-m3 with loss = 0.5*CE + 0.5*T^2*KL, T=2
# Caveat recorded in docs/adjustments.md: teacher logits on its own train set are
# near-one-hot, so most signal lives in the ~14k val rows + the dark knowledge.
#   Results: output/pat/ft_results_bgem3_distill.csv
#   Weights: output/pat/ft_BAAI__bge-m3_distill_qwen3/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

TEACHER=output/pat/ft_Qwen__Qwen3-Embedding-0.6B/checkpoint-10500
LOGITS=output/pat/teacher_logits_qwen3.npz

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

if [[ ! -f "${LOGITS}" ]]; then
  echo "dumping teacher logits from ${TEACHER}"
  "${PY}" -u -m src.dump_logits \
    --ckpt "${TEACHER}" --max_len 512 --batch_size 48 --out "${LOGITS}"
fi

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --distill_from "${LOGITS}" --distill_alpha 0.5 --distill_T 2.0 \
  --tag distill_qwen3 \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_bgem3_distill.csv"
