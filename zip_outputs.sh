#!/usr/bin/env bash
# Zip training run outputs for transfer/backup.
# Run on the training server, from the repo root (where output/ lives).
#
#   ./zip_outputs.sh              # logs + metrics + configs (NO heavy weights) — small
#   ./zip_outputs.sh --weights    # everything, including checkpoint .safetensors — large
#
set -euo pipefail

INCLUDE_WEIGHTS=0
[[ "${1:-}" == "--weights" ]] && INCLUDE_WEIGHTS=1

# Timestamped name so repeated zips don't clobber each other.
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="training_outputs_${STAMP}.zip"

# Collect task-spooler stdout logs from the 4 per-GPU queues into a staging dir,
# so they ride along in the zip (ts stores them in /tmp and would otherwise be lost).
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/ts_logs"
for g in 0 1 2 3; do
  sock="/tmp/ts.gpu$g"
  [[ -S "$sock" ]] || continue
  # For each finished job, copy its captured output file, named by GPU+jobid.
  TS_SOCKET="$sock" tsp 2>/dev/null | awk 'NR>1 {print $1}' | while read -r id; do
    f=$(TS_SOCKET="$sock" tsp -o "$id" 2>/dev/null) || continue
    [[ -f "$f" ]] && cp "$f" "$STAGE/ts_logs/gpu${g}_job${id}.log"
  done
done

# Build the zip.
if [[ $INCLUDE_WEIGHTS -eq 1 ]]; then
  echo ">> including checkpoint weights (large)"
  zip -r "$OUT" output/ "$STAGE/ts_logs" -x '*.tmp'
else
  echo ">> logs + metrics + configs only (excluding *.safetensors / optimizer states)"
  zip -r "$OUT" output/ "$STAGE/ts_logs" \
    -x '*.safetensors' '*.bin' '*optimizer*' '*.pt' '*scheduler*' '*.tmp'
fi

echo ">> wrote $OUT ($(du -h "$OUT" | cut -f1))"
