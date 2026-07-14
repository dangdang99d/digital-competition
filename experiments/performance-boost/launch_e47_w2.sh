#!/usr/bin/env bash
# E47 wave-2: DropHead p-sweep + DropHead×MSD stacks. 14k-val (56k/14k), 8 epochs
# (higher-p regularization peaks later than p=0.1's ep5). Arg $1 = gpu list "inst1"/"inst2".
cd /workspace/repo || exit 1
export PYTHONPATH=.
PY=/venv/main/bin/python
M=ibm-granite/granite-embedding-311m-multilingual-r2
BASE="--model $M --serialize richargs --loss ls --label_smoothing 0.1365 --precision bf16 \
--init_seed 42 --group_by_length --epochs 8 --max_len 512 --save_dtype fp16 --lr 3.466e-05 \
--batch_size 16 --grad_accum 1 --warmup_ratio 0.1441 --weight_decay 1.7158e-03 \
--awp_gamma 2.0116e-03 --awp_lr 2.3090e-04 --awp_start_epoch 1.0 --out_dir output/e47"
mkdir -p /workspace/logs
run() { local gpu=$1 tag=$2; shift 2
  setsid nohup bash -c "cd /workspace/repo; export PYTHONPATH=.; \
    CUDA_VISIBLE_DEVICES=$gpu $PY -m src.finetune $BASE \
    --results_name ft_e47_${tag}.csv --tag e47_${tag} $* \
    > /workspace/logs/e47_${tag}.log 2>&1; touch /workspace/logs/e47_${tag}.DONE" >/dev/null 2>&1 &
  echo "launched $tag on gpu $gpu (pid $!)"
}
if [ "$1" = "inst1" ]; then
  run 0 dh005      --drophead_p 0.05
  run 1 dh015      --drophead_p 0.15
  run 2 dh020      --drophead_p 0.20
  run 3 dh025      --drophead_p 0.25
elif [ "$1" = "inst2" ]; then
  run 0 dh010_msd  --drophead_p 0.10 --msd_k 5 --msd_p 0.3
  run 1 dh015_msd  --drophead_p 0.15 --msd_k 5 --msd_p 0.3
  run 3 dh010_8ep  --drophead_p 0.10
fi
echo "wave-2 $1 launched"
