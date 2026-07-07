#!/bin/bash
# E16 — qwen3 depth-prune 28->14 (ShortGPT BI selection) + recovery-FT
# Mirrors the E4 recovery recipe: full_ft, linear head, 2ep @ lr 5e-6, richargs, full_data, ce.
set -euo pipefail
cd /home/ocean/dacon
KEEP=$(cat experiments/pruning/e16_keep.txt)
echo "keep_layer_idx = ${KEEP}"
export CUDA_VISIBLE_DEVICES=3
PY=/home/ocean/miniconda3/envs/dacon/bin/python
"${PY}" -u -m src.finetune \
  --model Qwen/Qwen3-Embedding-0.6B \
  --init_from ./output/pat/ft_Qwen__Qwen3-Embedding-0.6B_e8b_qwen3_richargs_full/checkpoint-8314 \
  --keep_layer_idx "${KEEP}" \
  --serialize richargs \
  --full_data \
  --loss ce \
  --epochs 2 \
  --lr 5e-6 \
  --max_len 512 \
  --batch_size 4 \
  --grad_accum 4 \
  --tag e16_qwen3_depth14_recover \
  --out_dir ./output/pat \
  --results_name ft_results_e16.csv
