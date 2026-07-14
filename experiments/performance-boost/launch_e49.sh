#!/usr/bin/env bash
# E49 phase-2: DropHead-family variants on the 5090 8-GPU box. 14k-val screen (56k/14k),
# 8 epochs, one arm per GPU. Baseline for the p-dependent arms = wave-1 constant-p=0.1
# DropHead 0.7826 (matched p → isolates the variant axis). AWP stays (deployment recipe).
cd /workspace/repo || exit 1
export PYTHONPATH=. HF_HOME=/workspace/.hf_home
PY=/venv/main/bin/python
M=ibm-granite/granite-embedding-311m-multilingual-r2
BASE="--model $M --serialize richargs --loss ls --label_smoothing 0.1365 --precision bf16 \
--init_seed 42 --group_by_length --epochs 8 --max_len 512 --save_dtype fp16 --lr 3.466e-05 \
--batch_size 16 --grad_accum 1 --warmup_ratio 0.1441 --weight_decay 1.7158e-03 \
--awp_gamma 2.0116e-03 --awp_lr 2.3090e-04 --awp_start_epoch 1.0 --out_dir output/e49"
mkdir -p /workspace/logs
run() { local gpu=$1 tag=$2; shift 2
  setsid nohup bash -c "cd /workspace/repo; export PYTHONPATH=. HF_HOME=/workspace/.hf_home; \
    CUDA_VISIBLE_DEVICES=$gpu $PY -m src.finetune $BASE \
    --results_name ft_e49_${tag}.csv --tag e49_${tag} $* \
    > /workspace/logs/e49_${tag}.log 2>&1; touch /workspace/logs/e49_${tag}.DONE" >/dev/null 2>&1 &
  echo "launched $tag on gpu $gpu (pid $!)"
}
run 0 s1_updown  --drophead_p 0.1 --drophead_schedule updown
run 1 s2_warmup  --drophead_p 0.1 --drophead_schedule warmup
run 2 s3_layerdrop --layerdrop_p 0.1
run 3 s4_dh_ld   --drophead_p 0.1 --layerdrop_p 0.1
run 4 s5_dropffn --dropffn_p 0.1
run 5 s6_token   --drophead_p 0.1 --drophead_granularity token
run 6 s7_corr    --drophead_p 0.1 --drophead_correlated
run 7 s8_ramp    --drophead_p 0.1 --drophead_layer_ramp
echo "E49 phase-2 launched (8 arms)"
