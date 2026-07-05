#!/bin/bash
#SBATCH --job-name=bgem3-histdrop
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=sbatch/logs/%x-%j.out

# TARGET MODEL: BAAI/bge-m3

# History dropout p=0.2: each history event is dropped independently, RE-DRAWN
# every epoch (dynamic serialization, val untouched). Sibling of the hist
# ablation: instead of a hard cap, it forces the model not to rely on any single
# history line — a robustness/content experiment. Compare against hist-abl-pat
# task 0 (identical recipe, p=0).
#   Results: output/pat/ft_results_bgem3_histdrop.csv
#   Weights: output/pat/ft_BAAI__bge-m3_histdrop02/

set -euo pipefail

cd /NAS2/ksjang0515/dacon
mkdir -p sbatch/logs output/pat

echo "job ${SLURM_JOB_ID} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

PY=/NAS2/ksjang0515/home/miniconda3/envs/dacon/bin/python
export HF_HOME=/NAS2/ksjang0515/hf_cache

"${PY}" -u -m src.finetune \
  --model BAAI/bge-m3 \
  --hist_dropout 0.2 --tag histdrop02 \
  --max_len 1024 --epochs 3 --lr 2e-5 \
  --batch_size 4 --grad_accum 4 \
  --out_dir ./output/pat \
  --results_name "ft_results_bgem3_histdrop.csv"
