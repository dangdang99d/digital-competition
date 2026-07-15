#!/bin/bash
# E39 C-MLM master: ensure XLM-R cached → run 10-shard × {balanced,blanket} fleet across 4 GPUs
# → merge shards. Robust one-shot: launch once detached, poll /workspace/aug/CMLM_ALL_DONE.
set -e
cd /workspace/repo
# XLM-R was rsynced from local into the cache → run FULLY OFFLINE (box HF is rate-limited).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/aug

echo "[1/3] XLM-R (rsynced, offline)"
/venv/main/bin/python -c "from transformers import AutoModelForMaskedLM; AutoModelForMaskedLM.from_pretrained('FacebookAI/xlm-roberta-large'); print('XLMR loads OK')"

echo "[2/3] launch fleet (10 shards x 2 targetings, 2 threads/worker)"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
N=10
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

echo "[3/3] merge shards"
cat /workspace/aug/cmlm_bal_s*.jsonl > /workspace/aug/cmlm_balanced.jsonl
cat /workspace/aug/cmlm_bln_s*.jsonl > /workspace/aug/cmlm_blanket.jsonl
echo "balanced=$(wc -l < /workspace/aug/cmlm_balanced.jsonl) blanket=$(wc -l < /workspace/aug/cmlm_blanket.jsonl)" > /workspace/aug/CMLM_ALL_DONE
