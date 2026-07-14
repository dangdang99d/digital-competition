#!/bin/bash
# E46 arm A — wait for native sweep ALL_DONE, then build+run TRT engines sequentially.
REPO=/mnt/nfs/data/research/ocean_backup-dacon
SWEEPLOG=/tmp/claude-1000/-mnt-nfs-data-research-ocean-backup-dacon/6871b956-0eef-4496-b132-386bd9217dc3/scratchpad/native_sweep3.log
OUT=/tmp/claude-1000/-mnt-nfs-data-research-ocean-backup-dacon/6871b956-0eef-4496-b132-386bd9217dc3/scratchpad/trt_chain.log

echo "waiting for native sweep ALL_DONE..." >> "$OUT"
until grep -q "^ALL_DONE" "$SWEEPLOG" 2>/dev/null; do
  # bail if sweep died without finishing
  if ! ps -eo args | grep -q "[c]onda/envs/dacon/bin/python $REPO/experiments/compression/quantization/run_all_native.py" \
     && ! ps -eo args | grep -q "[c]onda/envs/dacon/bin/python experiments/compression/quantization/run_all_native.py"; then
    grep -q "^ALL_DONE" "$SWEEPLOG" 2>/dev/null || { echo "SWEEP_DIED_BEFORE_DONE — starting TRT anyway (GPU free)" >> "$OUT"; break; }
  fi
  sleep 15
done

for v in fp32 fp16 int8_entropy int8_minmax; do
  echo "=== TRT $v ===" >> "$OUT"
  docker run --rm --gpus all -v "$REPO":/ws nvcr.io/nvidia/tensorrt:24.12-py3 \
    python /ws/experiments/compression/quantization/trt_build_infer.py "$v" >> "$OUT" 2>&1
  echo "=== TRT $v exit=$? ===" >> "$OUT"
done
echo "TRT_CHAIN_DONE" >> "$OUT"
