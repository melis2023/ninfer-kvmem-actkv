#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git cmake ninja-build build-essential pkg-config aria2 curl python3-venv ccache \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev \
    libswresample-dev libcurl4-openssl-dev
# Python 3.11 is selected explicitly, independently of the CUDA image's default interpreter.
curl -LsSf https://astral.sh/uv/install.sh | sh
/root/.local/bin/uv python install 3.11
/root/.local/bin/uv venv --python 3.11 /workspace/ninfer-work/py311
/root/.local/bin/uv pip install --python /workspace/ninfer-work/py311/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch
/root/.local/bin/uv pip install --python /workspace/ninfer-work/py311/bin/python numpy safetensors gguf 'pytest==9.1.1' \
    'huggingface_hub[hf_xet]'
/workspace/ninfer-work/py311/bin/python --version
nvcc --version
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader
/workspace/ninfer-work/py311/bin/python scripts/pods/host_gate.py
