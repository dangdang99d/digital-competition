#!/usr/bin/env bash
# Granite (E31) compression wave-1 — run ON the granite box (/workspace/repo).
# Baseline + FFN low-rank SVD probe (training-free) + FFN-neuron width-prune recovery arms,
# all vs the t031 baseline on the honest 3.5k held-out. Quantization is handled elsewhere;
# this covers the FFN low-rank + structured-width axes for granite.
set -e
cd /workspace/repo
PY=/venv/main/bin/python
G=/workspace/t031/model/granite-311m-e8a-ls-awp-t031      # t031 baseline (LB 0.79300)
OUT=output/e31g
mkdir -p "$OUT" logs

# GPU0: baseline logits + training-free FFN low-rank SVD-degradation curve
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
  $PY -m experiments.compression.eval_compress --model_dir $G --out $OUT/base.npz
  $PY -m analysis.palu_probe_ffn --model_dir $G --proj ffn --n_val 3000 --ratios 1.0,0.875,0.75,0.625,0.5,0.375,0.25 --out_json $OUT/ffn_svd_curve.json
  touch $OUT/DONE_probe
" > logs/e31g_probe.log 2>&1 &
echo "GPU0: baseline + FFN SVD probe launched"

# GPU1/2/3: FFN-neuron width prune (ModernBERT GeGLU) + 2ep recovery
launch_ffn () {  # $1=gpu $2=keep $3=tag
  CUDA_VISIBLE_DEVICES=$1 nohup $PY -m src.finetune \
    --model ibm-granite/granite-embedding-311m-multilingual-r2 \
    --serialize richargs --full_data --loss ls --epochs 2 --lr 1e-5 \
    --init_from "$G" --ffn_keep "$2" --tag "e31_t031_$3" \
    > "logs/e31g_$3.log" 2>&1 &
  echo "GPU$1: FFN-width keep-$2 ($3)"
}
launch_ffn 1 0.75 ffn75
launch_ffn 2 0.50 ffn50
launch_ffn 3 0.25 ffn25
echo "granite wave-1 launched"
