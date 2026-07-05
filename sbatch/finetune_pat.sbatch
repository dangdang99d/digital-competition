#!/bin/bash
#SBATCH --job-name=fullft-qwen3
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodelist=pat-g2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: Qwen/Qwen3-Embedding-0.6B

# Qwen3-Embedding-0.6B FULL fine-tune on pat-g2 (rtx4090, 24GB) with AdamW —
# the decisive rerun: its 0.5227 on moana was an SGD artifact (the 2080 Ti
# couldn't fit Adam). SGD costs ~37 F1 pts (granite: 0.7153 AdamW vs 0.3439 SGD).
# Beat 0.7153 (granite AdamW) and Qwen3 becomes the submission model.
#   Submit:  sbatch sbatch/finetune_pat.sbatch
#   Log:     sbatch/logs/fullft-qwen3-<jobid>.out
#   Results: output/pat/ft_results_qwen3.csv  (+ checkpoint under output/pat/)
# Everything goes under output/pat/ so nothing collides with moana's output/
# (same-named checkpoint dir ft_Qwen__Qwen3-Embedding-0.6B) when rsynced locally.

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output

MODEL="Qwen/Qwen3-Embedding-0.6B"

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
echo "model: ${MODEL}"
nvidia-smi -L || true

# absolute env python — no conda activation needed inside batch jobs (pat quirk).
# The env is created/updated MANUALLY from an interactive shell (see `commands`),
# never from inside a batch job.
PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

# AdamW (default optim), same effective batch 16 as the granite baseline.
# 24GB allows batch 8 x accum 2 (faster than 4x4 on the 2080 Ti).
"${PY}" -u -m src.finetune \
  --model "${MODEL}" --epochs 3 --out_dir ./output/pat \
  --batch_size 8 --grad_accum 2 --lr 2e-5 \
  --results_name "ft_results_qwen3.csv"
