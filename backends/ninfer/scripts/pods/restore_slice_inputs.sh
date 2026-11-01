#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
export HF_TOKEN_PATH=/run/ninfer-hf/token
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
trap 'rm -f -- /run/ninfer-hf/token' EXIT
/root/.local/bin/uv pip install --python "$root/py311/bin/python" 'huggingface_hub==1.17.0' hf-xet
"$root/py311/bin/python" "$NINFER_JOB_DIR/inputs/restore_slice_inputs.py"
