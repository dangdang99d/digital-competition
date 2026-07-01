#!/bin/bash
#SBATCH --job-name=dacon
#SBATCH --partition=batch
#SBATCH --qos=normal
##SBATCH --nodelist=pat-t3          # uncomment to pin to a specific node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

set -euo pipefail

# --- project ---
cd /NAS2/ksjang0515/dacon

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

# --- run ---
# Call the env's python by absolute path (like line_ct) so no conda activation is
# needed inside the batch job. Adjust the miniconda path / env name if they differ.
# If you ever need conda on an interactive shell instead:
#   source /NAS2/ksjang0515/home/config/bash/.bashrc
#   source /NAS2/ksjang0515/home/config/bash/.bash_profile
#   conda activate dacon
PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python

# Edit this to point at whatever you want to run.
"${PY}" -u baseline/script.py
