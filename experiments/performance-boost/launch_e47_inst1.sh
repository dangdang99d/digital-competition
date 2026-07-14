#!/usr/bin/env bash
# E47 wave-1 instance-1 (4x3090): anchor + A1 neg-LS + A2 grouped-LS + A3 drophead.
# 56k/14k split (NO --full_data), 6 epochs, per-epoch 14k-val macro-F1, one run per GPU.
cd /workspace/repo || exit 1
export PYTHONPATH=.
PY=/venv/main/bin/python
M=ibm-granite/granite-embedding-311m-multilingual-r2
BASE="--model $M --serialize richargs --precision bf16 --init_seed 42 --group_by_length \
--epochs 6 --max_len 512 --save_dtype fp16 --lr 3.466e-05 --batch_size 16 --grad_accum 1 \
--warmup_ratio 0.1441 --weight_decay 1.7158e-03 --awp_gamma 2.0116e-03 --awp_lr 2.3090e-04 \
--awp_start_epoch 1.0 --out_dir output/e47"
mkdir -p /workspace/logs
run() { # gpu tag extra...
  local gpu=$1 tag=$2; shift 2
  setsid nohup bash -c "cd /workspace/repo; export PYTHONPATH=.; \
    CUDA_VISIBLE_DEVICES=$gpu $PY -m src.finetune $BASE \
    --results_name ft_e47_${tag}.csv --tag e47_${tag} $* \
    > /workspace/logs/e47_${tag}.log 2>&1; touch /workspace/logs/e47_${tag}.DONE" >/dev/null 2>&1 &
  echo "launched $tag on gpu $gpu (pid $!)"
}
run 0 anchor  --loss ls --label_smoothing 0.1365
run 1 negls   --loss lsmat --ls_matrix experiments/performance-boost/lsmat_neg01.npy
run 2 grpls   --loss lsmat --ls_matrix experiments/performance-boost/lsmat_grouped_soft.npy
run 3 drophd  --loss ls --label_smoothing 0.1365 --drophead_p 0.1
echo "INST1 wave-1 launched"
