#!/usr/bin/env bash
# E50 box B (44689856, 8x5090): fixed anchor cells x 3 init_seeds (split stays seed-42;
# --init_seed varies init/shuffle/dropout only). 15 runs, 2/GPU. Gives the seed-MEAN and
# seed-SPREAD for: AWP baseline (never seed-replicated!), DropHead const, layer-ramp,
# DropHead+MSD (k5 = the measured wave-2 cell), ramp+MSD. Epochs pinned 5, standard split.
cd /workspace/repo || exit 1
export PYTHONPATH=. HF_HOME=/workspace/.hf_home
PY=/venv/main/bin/python
M=ibm-granite/granite-embedding-311m-multilingual-r2
BASE="--model $M --serialize richargs --loss ls --precision bf16 --group_by_length \
--epochs 5 --max_len 512 --save_dtype fp16 --lr 3.466e-05 --batch_size 16 --grad_accum 1 \
--warmup_ratio 0.1441 --label_smoothing 0.1365 --weight_decay 1.7158e-03 \
--awp_gamma 2.0116e-03 --awp_lr 2.3090e-04 --awp_start_epoch 1.0 --out_dir output/e50a"
mkdir -p /workspace/logs
i=0
run() { local tag=$1; shift
  local gpu=$((i % 8)); i=$((i + 1))
  setsid nohup bash -c "cd /workspace/repo; export PYTHONPATH=. HF_HOME=/workspace/.hf_home; \
    CUDA_VISIBLE_DEVICES=$gpu $PY -m src.finetune $BASE \
    --results_name ft_e50a_${tag}.csv --tag e50a_${tag} $* \
    > /workspace/logs/e50a_${tag}.log 2>&1; touch /workspace/logs/e50a_${tag}.DONE" >/dev/null 2>&1 &
  echo "launched $tag on gpu $gpu (pid $!)"
}
for s in 42 7 13; do
  run a0_s${s}  --init_seed $s
  run d1_s${s}  --init_seed $s --drophead_p 0.1
  run d2_s${s}  --init_seed $s --drophead_p 0.1 --drophead_layer_ramp
  run d3_s${s}  --init_seed $s --drophead_p 0.1 --msd_k 5 --msd_p 0.3
  run d4_s${s}  --init_seed $s --drophead_p 0.1 --drophead_layer_ramp --msd_k 5 --msd_p 0.3
done
echo "E50 anchors launched: 15 runs (5 configs x 3 init_seeds), 2/GPU"
