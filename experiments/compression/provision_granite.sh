#!/usr/bin/env bash
# Run LOCALLY (repo root) once the granite box is READY. Ships my unpushed changes
# (origin lacks them) + the t031 baseline zip, then unzips + launches wave-1.
set -e
H=root@ssh7.vast.ai; P=18884; K=.vast/vast_ed25519
SSH="ssh -i $K -p $P $H -o StrictHostKeyChecking=accept-new"
SCP="scp -i $K -P $P -o StrictHostKeyChecking=accept-new"

# 1. ship modified source + harness (unpushed) — MUST overwrite origin-HEAD clone
$SCP src/finetune.py src/factored_ffn.py $H:/workspace/repo/src/
$SCP analysis/palu_probe_ffn.py $H:/workspace/repo/analysis/
$SCP experiments/compression/eval_compress.py experiments/compression/bi_probe.py \
     experiments/compression/granite_wave1.sh $H:/workspace/repo/experiments/compression/

# 2. ship + unzip the t031 baseline (659MB)
$SCP submissions/submit_0714_awp_t031.zip $H:/workspace/
$SSH 'mkdir -p /workspace/t031 && cd /workspace && unzip -o -q submit_0714_awp_t031.zip -d t031 && ls t031/model'

# 3. launch wave-1
$SSH 'cd /workspace/repo && bash experiments/compression/granite_wave1.sh'
echo "granite wave-1 dispatched"
