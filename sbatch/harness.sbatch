#!/bin/bash
#SBATCH --job-name=harness
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# Guardrail harness: prompt a reasoning LLM to classify the next action. The
# model thinks natively (<think> stage, Qwen3 enable_thinking) then answers;
# output is guardrailed to the 14 valid labels (parse+validate, logprob
# fallback). No fine-tuning — iterate on the markdown prompt (prompts/*.md).
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
# Thinking mode generates ~a few hundred tokens/sample at bs=1 — budget hours, not
# minutes, for 1000 samples. Use --no_thinking or --mode logprob for quick checks.
"${PY}" -u -m src.harness \
  --model Qwen/Qwen3-4B \
  --prompt prompts/default.md \
  --n_dev 1000 --shots 2 \
  --max_new_tokens 1024 \
  --out_dir ./output
