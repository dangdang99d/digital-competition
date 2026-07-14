#!/usr/bin/env bash
# E47 wave-1 instance-2 (4x3090): A4 child-tuning + A5 msd head (2 GPUs; GPU2/3 held
# for wave-1-informed refinements / the winner's full_data retrain).
cd /workspace/repo || exit 1
export PYTHONPATH=.
PY=/venv/main/bin/python
M=ibm-granite/granite-embedding-311m-multilingual-r2
BASE="--model $M --serialize richargs --precision bf16 --init_seed 42 --group_by_length \
--epochs 6 --max_len 512 --save_dtype fp16 --lr 3.466e-05 --batch_size 16 --grad_accum 1 \
--warmup_ratio 0.1441 --label_smoothing 0.1365 --weight_decay 1.7158e-03 \
--awp_gamma 2.0116e-03 --awp_lr 2.3090e-04 --awp_start_epoch 1.0 --out_dir output/e47"
mkdir -p /workspace/logs
run() { local gpu=$1 tag=$2; shift 2
  setsid nohup bash -c "cd /workspace/repo; export PYTHONPATH=.; \
    CUDA_VISIBLE_DEVICES=$gpu $PY -m src.finetune $BASE \
    --results_name ft_e47_${tag}.csv --tag e47_${tag} $* \
    > /workspace/logs/e47_${tag}.log 2>&1; touch /workspace/logs/e47_${tag}.DONE" >/dev/null 2>&1 &
  echo "launched $tag on gpu $gpu (pid $!)"
}
run 0 child  --loss ls --child_p 0.3 --child_fisher_batches 64
run 1 msd    --loss ls --msd_k 5 --msd_p 0.3
echo "INST2 wave-1 launched"
