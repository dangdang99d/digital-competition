#!/usr/bin/env bash
# vast_onstart.sh — instance bootstrap, run by the "dacon-3090" vast.ai template at
# every boot (the template curls this from GitHub raw, so edits here take effect on
# the next instance without recreating the template).
# Idempotent. Logs to /workspace/onstart.log. Touches /workspace/READY when usable.
exec > /workspace/onstart.log 2>&1
set -ex

BRANCH="${DACON_BRANCH:-research/token-selection}"

cd /workspace
if [ ! -d repo/.git ]; then
    git clone --depth 1 -b "$BRANCH" https://github.com/dangdang99d/digital-competition.git repo
fi

# torch is already in the image; its requirements pin has a +cu128 local suffix that
# pip treats as a different version and re-downloads 2.5GB — skip the line
grep -v '^torch' repo/requirements.txt > /tmp/reqs.txt
pip install --no-cache-dir -r /tmp/reqs.txt

echo 'export TQDM_DISABLE=1' >> /root/.bashrc

touch /workspace/READY
