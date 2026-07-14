#!/usr/bin/env bash
# E47 wiring smoke — 64 samples, 1 epoch, bs4, max_len 128, AWP ON (tests composition).
# Each new lever must train + eval + save without error. Run from anywhere; ~2 min/arm on GPU.
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-/home/kyusang/.conda/envs/dacon/bin/python}
BASE="--model ibm-granite/granite-embedding-311m-multilingual-r2 --serialize richargs \
 --loss ls --precision bf16 --epochs 1 --max_len 128 --batch_size 4 --limit 64 \
 --awp_gamma 2e-3 --awp_lr 2.3e-4 --awp_start_epoch 0 \
 --out_dir output/e47_smoke --results_name ft_e47_smoke.csv"
PYTHONPATH=. $PY -m src.finetune $BASE --tag smoke_lsneg \
  --loss lsmat --ls_matrix experiments/performance-boost/lsmat_neg01.npy
PYTHONPATH=. $PY -m src.finetune $BASE --tag smoke_lsgrp \
  --loss lsmat --ls_matrix experiments/performance-boost/lsmat_grouped_soft.npy
PYTHONPATH=. $PY -m src.finetune $BASE --tag smoke_drophead --drophead_p 0.3
PYTHONPATH=. $PY -m src.finetune $BASE --tag smoke_child --child_p 0.3 --child_fisher_batches 2
PYTHONPATH=. $PY -m src.finetune $BASE --tag smoke_msd --msd_k 3 --msd_p 0.3
echo "E47 SMOKE ALL OK"
