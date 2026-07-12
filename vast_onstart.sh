#!/usr/bin/env bash
# vast_onstart.sh — instance bootstrap, run by the "dacon-3090" vast.ai template at
# every boot (the template curls this from GitHub raw, so edits here take effect on
# the next instance without recreating the template).
# Idempotent. Logs to /workspace/onstart.log. Touches /workspace/READY when usable.
exec > /workspace/onstart.log 2>&1
set -ex

BRANCH="${DACON_BRANCH:-research/token-selection}"

# vast's own images keep python in a venv — activate it if present
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate

# runtime images (e.g. pytorch/pytorch:latest) lack a C toolchain, which
# torch.compile/triton needs at run time
command -v gcc >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq build-essential; }

cd /workspace
if [ ! -d repo/.git ]; then
    git clone --depth 1 -b "$BRANCH" https://github.com/dangdang99d/digital-competition.git repo
fi

# torch is already in the image; its requirements pin has a +cu128 local suffix that
# pip treats as a different version and re-downloads 2.5GB — skip the line
grep -v '^torch' repo/requirements.txt > /tmp/reqs.txt
# hosts with a stale cached pytorch:latest ship python 3.10; sklearn 1.8 needs >=3.11.
# 1.7.2 is metric-identical for our use (macro-F1 only)
python -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' || \
    sed -i 's/scikit-learn==1.8.0/scikit-learn==1.7.2/' /tmp/reqs.txt
pip install --no-cache-dir -r /tmp/reqs.txt

echo 'export TQDM_DISABLE=1' >> /root/.bashrc

# record what this host's cached image actually gave us (floating :latest tag)
python -c "import sys, torch, sklearn; print(f'VERSIONS python={sys.version.split()[0]} torch={torch.__version__} sklearn={sklearn.__version__}')"

touch /workspace/READY
