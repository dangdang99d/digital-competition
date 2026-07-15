#!/bin/bash
# E39 LLM generation: Qwen2.5-14B-Instruct-AWQ (rsynced, offline) via vLLM, sharded across
# all 4 GPUs — balanced on GPU0/1, blanket on GPU2/3 (2 shards each). Each instance loads the
# 14B AWQ (~10GB, fits one 3090). Chunked/incremental writes. Launch in tmux; poll LLM_ALL_DONE.
cd /workspace/repo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/aug
PY=/venv/vllm/bin/python

run() {  # targeting, pre, gpu, shard
  CUDA_VISIBLE_DEVICES=$3 $PY -m experiments.word_aug.gen_llm \
    --targeting "$1" --shard "$4" --nshards 2 \
    --out "/workspace/aug/llm_${2}_s${4}.jsonl" > "/workspace/aug/llm_${2}_s${4}.log" 2>&1 &
}
run balanced bal 0 0
run balanced bal 1 1
run blanket  bln 2 0
run blanket  bln 3 1
wait

cat /workspace/aug/llm_bal_s*.jsonl > /workspace/aug/llm_balanced.jsonl
cat /workspace/aug/llm_bln_s*.jsonl > /workspace/aug/llm_blanket.jsonl
echo "balanced=$(wc -l < /workspace/aug/llm_balanced.jsonl) blanket=$(wc -l < /workspace/aug/llm_blanket.jsonl)" > /workspace/aug/LLM_ALL_DONE
