#!/bin/bash
# E39 arm B3 — Qwen3-30B-A3B-Instruct-2507-FP8 paraphrase fleet on the 8×5090 box.
# Order of operations (each a hard gate against the prior AWQ-garbage failure):
#   0. free-GPU guard — REFUSE any GPU in PAIRS that is not actually idle (protects the E35 job)
#   1. stack_check    — GPUs sm_120 + torch/CUDA + vLLM are FP8-capable (no model load)
#   2. smoke gate     — load model on ONE pair, 8 probes, ASSERT sane output; ABORT fan-out on fail
#   3. fan out        — 3 shards × TP=2 across the FREE pairs (0,1)(2,3)(4,5), per targeting
#   4. merge          — concat shard files -> qwen_{balanced,blanket}.jsonl
# GPUs 6,7 are RUNNING the E35 AWP job (2026-07-14) — this box shares them; we use only 0–5.
# FP8 fails on this box's vLLM? re-run with DTYPE=bfloat16 (30B bf16 ~61GB fits TP=2). No quant kernel.
#
#   MODEL=/workspace/models/Qwen3-30B-A3B-Instruct-2507-FP8 bash experiments/word_aug/run_qwen_fleet.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-/workspace/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
DTYPE="${DTYPE:-auto}"                 # auto = honor FP8 checkpoint; bfloat16 = no-quant fallback
TP="${TP:-2}"                          # 30GB FP8 needs >1 GPU (32GB card can't hold weights+KV)
OUTDIR="${OUTDIR:-output/e39/qwen_local}"
# Only the 6 FREE GPUs (0–5); 6,7 belong to the E35 job. Override via PAIRS_STR="0,1 2,3 ...".
read -r -a PAIRS <<< "${PAIRS_STR:-0,1 2,3 4,5}"
NSHARDS=${#PAIRS[@]}
PY="${PY:-python}"
mkdir -p "$OUTDIR"

echo "=== [0/4] free-GPU guard (never stomp another job) ==="
for pair in "${PAIRS[@]}"; do
    for g in ${pair//,/ }; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
        if [ "${used:-9999}" -gt 1000 ]; then
            echo "ABORT: GPU $g has ${used}MiB used — NOT free. Another job (E35?) may be on it."
            echo "       Re-check with nvidia-smi and set PAIRS_STR to only-idle pairs."
            exit 1
        fi
        echo "  GPU $g: ${used}MiB used — free ✓"
    done
done

echo "=== [1/4] stack check ==="
$PY -m experiments.word_aug.stack_check || { echo "STACK CHECK FAILED — aborting"; exit 1; }

echo "=== [2/4] smoke gate (model=$MODEL dtype=$DTYPE tp=$TP) ==="
CUDA_VISIBLE_DEVICES="${PAIRS[0]}" $PY -m experiments.word_aug.gen_llm_vllm \
    --smoke --model "$MODEL" --dtype "$DTYPE" --tp "$TP" \
    2>&1 | tee "$OUTDIR/smoke.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "SMOKE GATE FAILED — NOT fanning out. Inspect $OUTDIR/smoke.log."
    echo "If FP8-related, retry: DTYPE=bfloat16 TP=2 bash $0"
    exit 1
fi
echo "smoke passed — proceeding to full generation"

for TARGET in balanced blanket; do
    echo "=== [3/4] generate targeting=$TARGET ($NSHARDS shards) ==="
    pids=()
    for i in "${!PAIRS[@]}"; do
        CUDA_VISIBLE_DEVICES="${PAIRS[$i]}" $PY -m experiments.word_aug.gen_llm_vllm \
            --targeting "$TARGET" --out "$OUTDIR/${TARGET}_shard${i}.jsonl" \
            --model "$MODEL" --dtype "$DTYPE" --tp "$TP" \
            --shard "$i" --nshards "$NSHARDS" \
            > "$OUTDIR/${TARGET}_shard${i}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    if [ "$fail" -ne 0 ]; then
        echo "a $TARGET shard failed — see $OUTDIR/${TARGET}_shard*.log"; exit 1
    fi
    echo "=== [4/4] merge $TARGET shards ==="
    cat "$OUTDIR/${TARGET}_shard"*.jsonl > "$OUTDIR/qwen_${TARGET}.jsonl"
    echo "  -> $OUTDIR/qwen_${TARGET}.jsonl ($(wc -l < "$OUTDIR/qwen_${TARGET}.jsonl") rows)"
done

echo "DONE: $OUTDIR/qwen_balanced.jsonl + qwen_blanket.jsonl"
