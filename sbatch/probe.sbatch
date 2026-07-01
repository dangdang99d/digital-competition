#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# Frozen-backbone linear-probe sweep over pretrained encoders (model selection).
# Moana cluster. No --account/--qos needed (defaults work for this partition).
#   Submit:  sbatch sbatch/probe.sbatch
#   Logs:    sbatch/logs/probe-<jobid>.out
#   Results: output/probe_results.csv  (rsync back to local)

set -euo pipefail

cd /data/ksjang0515/dacon
mkdir -p sbatch/logs output

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/data/ksjang0515/conda/envs/dacon/bin/python

# Sweep all encoders in src/models.yaml with the full serialized context.
# HF_HOME keeps model downloads on the data volume (not the home quota).
export HF_HOME=/data/ksjang0515/hf_cache

"${PY}" -u -m src.probe --models src/models.yaml --input context --out_dir ./output
