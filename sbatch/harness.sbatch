#!/bin/bash
#SBATCH --job-name=harness
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# Guardrail harness: prompt a code LLM to classify the next action, constrained
# to the 14 valid labels (rank-by-logprob). No fine-tuning — iterate on the
# markdown prompt (prompts/*.md), not the code.
# Moana cluster. No --account/--qos needed.
#   Submit:  sbatch sbatch/harness.sbatch
#   Results: output/harness_results.csv

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python
export HF_HOME=/data/ksjang0515/hf_cache

# Iterate on the dev subset (fast). Add --full for the final number on all 14k val.
"${PY}" -u -m src.harness \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --prompt prompts/default.md \
  --n_dev 1000 --shots 2 \
  --out_dir ./output
