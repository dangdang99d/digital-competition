#!/bin/bash
# E39 C-MLM data-parallel fleet: N shards × {balanced, blanket} across 4 GPUs.
# Thread-capped (2/worker → ~40 threads total on 40 vCPU) + staggered starts so 20
# XLM-R loads don't thread-explode. HF_HUB_OFFLINE (model already cached) → no rate limit.
cd /workspace/repo
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
N=${N:-10}
mkdir -p /workspace/aug
for t in balanced blanket; do
  if [ "$t" = "balanced" ]; then pre=bal; else pre=bln; fi
  for i in $(seq 0 $((N-1))); do
    g=$((i % 4))
    CUDA_VISIBLE_DEVICES=$g /venv/main/bin/python -m experiments.word_aug.gen_cmlm \
      --targeting "$t" --shard "$i" --nshards "$N" \
      --out "/workspace/aug/cmlm_${pre}_s${i}.jsonl" \
      > "/workspace/aug/cmlm_${pre}_s${i}.log" 2>&1 &
    sleep 1
  done
done
wait
echo DONE > /workspace/aug/cmlm.DONE
