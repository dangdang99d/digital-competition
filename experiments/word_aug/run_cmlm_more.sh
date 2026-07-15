#!/bin/bash
# E39 C-MLM SECOND independent set (seed 43 → distinct target selection + distinct paraphrases).
# Gives 2x augmented data available (v2 seed42 + this seed43) to test volume effects.
# Same fixed generator (mask_p 0.10, garbage-filtered). Launch in tmux; poll CMLM_MORE_DONE.
cd /workspace/repo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p /workspace/aug
N=10
for t in balanced blanket; do
  if [ "$t" = "balanced" ]; then pre=bal; else pre=bln; fi
  for i in $(seq 0 $((N-1))); do
    g=$((i % 4))
    CUDA_VISIBLE_DEVICES=$g /venv/main/bin/python -m experiments.word_aug.gen_cmlm \
      --targeting "$t" --shard "$i" --nshards "$N" --seed 43 \
      --out "/workspace/aug/cmlm_${pre}_b_s${i}.jsonl" \
      > "/workspace/aug/cmlm_${pre}_b_s${i}.log" 2>&1 &
    sleep 1
  done
done
wait
cat /workspace/aug/cmlm_bal_b_s*.jsonl > /workspace/aug/cmlm_balanced_b.jsonl
cat /workspace/aug/cmlm_bln_b_s*.jsonl > /workspace/aug/cmlm_blanket_b.jsonl
echo "balanced_b=$(wc -l < /workspace/aug/cmlm_balanced_b.jsonl) blanket_b=$(wc -l < /workspace/aug/cmlm_blanket_b.jsonl)" > /workspace/aug/CMLM_MORE_DONE
