#!/bin/bash
#SBATCH --job-name=dacon
#SBATCH --partition=batch_ugrad_advisor_x
##SBATCH --nodelist=                 # uncomment + set to pin to a specific node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# Template sbatch for the DACON project on the Moana cluster.
# Copy per experiment; edit job-name, resources, and the run line.
# No --account/--qos needed (defaults work for partition batch_ugrad_advisor_x).
#
# Before submitting:
#   - check partitions/nodes:  sinfo
#   - ensure log dir exists:   mkdir -p sbatch/logs
#
# Submit:  sbatch sbatch/template_moana.sbatch

set -euo pipefail

# --- project ---
cd /data/ksjang0515/dacon

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

# --- run ---
# Call the env's python by absolute path so no conda activation is needed in the job.
# If you ever need conda on an interactive shell instead:
#   source /data/ksjang0515/config/.bashrc
#   source /data/ksjang0515/config/.profile
#   conda activate dacon
PY=/data/ksjang0515/conda/envs/dacon/bin/python

# Edit this to point at whatever you want to run.
"${PY}" -u baseline/script.py
