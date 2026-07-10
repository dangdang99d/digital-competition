#!/bin/bash
# Modified granite_ls × teammate protocol (2026-07-09).
#   FROM HIS RECIPE: --serialize names (his format, byte-identical port)
#                    --session_fold k  (StratifiedGroupKFold(5) by session, leakage-free)
#                    --warmup_ratio 0.1
#   KEPT FROM OURS:  LS eps=0.1, bf16 (auto on 3090), lr 2e-5, eff. batch 16 (4x4),
#                    max_len 512, 3 epochs, seed 42, best-epoch by val macro-F1.
# Baselines to read against: E21 names+full_data 0.7731 / E8a richargs+full_data 0.7803
#   (both on the LEAKY 3.5k slice — this run's val is a clean 14k session-held-out fold,
#    so expect a LOWER number; compare against his fold scores 0.7606/0.7676, not ours).
# Usage: bash sbatch/granite_names_ls_sessionfold.sh <fold 0-4> [gpu=0] [serialize=names]
#   fold 0 first; fold 1 second reproduces his 2-fold protocol.
#   3rd arg swaps the serializer arm: names | names_files | richfiles | richargs
set -euo pipefail
cd /home/ocean/dacon
FOLD="${1:?usage: $0 <fold 0-4> [gpu] [serialize]}"
GPU="${2:-0}"
SER="${3:-names}"
mkdir -p sbatch/logs output/pat

CUDA_VISIBLE_DEVICES="$GPU" nohup python -m src.finetune \
  --model ibm-granite/granite-embedding-311m-multilingual-r2 \
  --serialize "$SER" \
  --session_fold "$FOLD" \
  --warmup_ratio 0.1 \
  --loss ls --label_smoothing 0.1 \
  --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4 --max_len 512 --seed 42 \
  --tag "granite_${SER}_ls_sfold${FOLD}" \
  --out_dir ./output/pat \
  --results_name "ft_results_granite_${SER}_sfold.csv" \
  > "sbatch/logs/granite_${SER}_ls_sfold${FOLD}.out" 2>&1 &

echo "launched: serialize=${SER} fold=${FOLD} gpu=${GPU}"
echo "log:      sbatch/logs/granite_${SER}_ls_sfold${FOLD}.out"
