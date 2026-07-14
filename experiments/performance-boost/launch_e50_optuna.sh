#!/usr/bin/env bash
# E50 launcher. Usage: launch_e50_optuna.sh <off|on>
#   off = A1 (44900306): constant-p study, p[0.05,0.20], 16 workers 2/GPU
#   on  = A2 (44689856): layer-ramp study, p[0.08,0.30], 14 workers + 2 slots for the
#         a0 AWP-baseline seed runs (the yardstick; box B was cut to 2 boxes)
# Epochs pinned 5 · standard 56k/14k split · single-seed trials · SQLite local.
MODE=${1:?pass off|on}
cd /workspace/repo || exit 1
export PYTHONPATH=. HF_HOME=/workspace/.hf_home
PY=/venv/main/bin/python
$PY -c "import optuna" 2>/dev/null || $PY -m pip install -q optuna
mkdir -p /workspace/logs output/optuna
NW=16
if [ "$MODE" = "on" ]; then NW=14; fi
n=0
for g in 0 1 2 3 4 5 6 7; do
  for s in a b; do
    [ $n -ge $NW ] && break
    EXTRA=""
    [ $n -eq 0 ] && EXTRA="--enqueue_anchor"
    setsid nohup bash -c "cd /workspace/repo; export PYTHONPATH=. HF_HOME=/workspace/.hf_home; \
      CUDA_VISIBLE_DEVICES=$g $PY -u -m src.optuna_search --gpu_tag g${g}${s} \
      --n_trials 3 --study e50_ramp_${MODE} --db output/optuna/e50_${MODE}.db \
      --out_dir output/e50 --exp e50${MODE:0:2} --search_drophead --ramp_mode $MODE $EXTRA \
      > /workspace/logs/e50_worker_g${g}${s}.log 2>&1; \
      touch /workspace/logs/e50_worker_g${g}${s}.DONE" >/dev/null 2>&1 &
    echo "worker g${g}${s} pid $!"
    [ $n -eq 0 ] && sleep 25   # let worker 0 create the study + enqueue anchors
    n=$((n + 1))
  done
done
if [ "$MODE" = "on" ]; then
  # a0 AWP-baseline seed runs on the 2 spare slots (gpu 7): the yardstick's own seed spread
  M=ibm-granite/granite-embedding-311m-multilingual-r2
  BASE="--model $M --serialize richargs --loss ls --precision bf16 --group_by_length \
--epochs 5 --max_len 512 --save_dtype fp16 --lr 3.466e-05 --batch_size 16 --grad_accum 1 \
--warmup_ratio 0.1441 --label_smoothing 0.1365 --weight_decay 1.7158e-03 \
--awp_gamma 2.0116e-03 --awp_lr 2.3090e-04 --awp_start_epoch 1.0 --out_dir output/e50a"
  for seed in 42 7; do
    setsid nohup bash -c "cd /workspace/repo; export PYTHONPATH=. HF_HOME=/workspace/.hf_home; \
      CUDA_VISIBLE_DEVICES=7 $PY -m src.finetune $BASE --init_seed $seed \
      --results_name ft_e50a_a0_s${seed}.csv --tag e50a_a0_s${seed} \
      > /workspace/logs/e50a_a0_s${seed}.log 2>&1; touch /workspace/logs/e50a_a0_s${seed}.DONE" >/dev/null 2>&1 &
    echo "a0 baseline seed $seed pid $!"
  done
fi
echo "E50 $MODE launched"
