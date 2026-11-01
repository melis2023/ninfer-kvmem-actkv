#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates tmux python3-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
/root/.local/bin/uv venv --python 3.11 /workspace/ninfer-work/py311
/root/.local/bin/uv pip install --python /workspace/ninfer-work/py311/bin/python \
    'huggingface_hub==1.17.0' hf-xet pytest
/workspace/ninfer-work/py311/bin/python --version
curl -fsSL --max-time 30 -o /dev/null https://huggingface.co
df -h /workspace
free -m
