#!/bin/bash
# E25 — teammate-recipe repro on our pipeline (2026-07-09). TWO ARMS, one GPU each:
#   a  REPRO: everything at HIS settings that differ from ours —
#      --serialize names · CE (no LS) · --warmup_ratio 0.1 · --precision fp16
#      (his notebook: warmup_ratio 0.1, fp16 AMP, plain CE, names format)
#   b  STACK: his settings + our champion levers — + LS eps=0.1 + bf16
#   c  = b but serialization richmeta (OUR format with HIS full-path history style;
#        meta path style already shared) — direct test of his "richmeta>richargs" claim
# Both arms use --full_data (66.5k train / 3.5k val slice) per user instruction —
# stand-in for his train-on-all mode; NOT k-fold (--session_fold exists but unused here).
# Shared (identical in his recipe and ours): granite-311m-multilingual-r2, lr 2e-5,
# 3 epochs, effective batch 16 (4x4 == his 16x1), weight decay 0.01, max_len 512,
# hist cap 12, seed 42, best-epoch by val macro-F1.
# Read-outs (all on the same 3.5k slice):
#   E25a vs his LB 0.77427 / OOF base 0.7642  -> does our pipeline reproduce his recipe?
#   E25b - E25a                               -> what LS+bf16 add ON HIS recipe
#   E25b vs E21 names+LS 0.7731               -> warmup 0.1 vs 0.05, single axis
# Usage: bash sbatch/e25_miseo_recipe.sh <arm a|b|c> [gpu=0]
set -euo pipefail
cd /home/ocean/dacon
ARM="${1:?usage: $0 <a|b|c> [gpu]}"
GPU="${2:-0}"
mkdir -p sbatch/logs output/pat

SER="names"
if [ "$ARM" = "a" ]; then
  EXTRA=(--precision fp16)                              # CE is the default loss
  TAG="e25a_miseo_repro"
elif [ "$ARM" = "b" ]; then
  EXTRA=(--precision bf16 --loss ls --label_smoothing 0.1)
  TAG="e25b_miseo_ls_bf16"
elif [ "$ARM" = "c" ]; then
  EXTRA=(--precision bf16 --loss ls --label_smoothing 0.1)
  SER="richmeta"
  TAG="e25c_richmeta_ls_bf16"
else
  echo "arm must be a, b, or c" >&2; exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" nohup python -m src.finetune \
  --model ibm-granite/granite-embedding-311m-multilingual-r2 \
  --serialize "$SER" --full_data \
  --warmup_ratio 0.1 \
  "${EXTRA[@]}" \
  --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4 --max_len 512 --seed 42 \
  --tag "$TAG" \
  --out_dir ./output/pat --results_name ft_results_e25_miseo.csv \
  > "sbatch/logs/${TAG}.out" 2>&1 &

echo "launched E25${ARM} (${TAG}) on GPU ${GPU} -> sbatch/logs/${TAG}.out"
