#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
export HF_TOKEN_PATH=/run/ninfer-hf/token
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export OMP_NUM_THREADS=4
trap 'rm -f -- /run/ninfer-hf/token' EXIT
if ! "$root/py311/bin/python" -c 'import huggingface_hub, hf_xet' >/dev/null 2>&1; then
    /root/.local/bin/uv pip install --python "$root/py311/bin/python" 'huggingface_hub==1.17.0' hf-xet
fi
"$root/py311/bin/python" "$NINFER_JOB_DIR/inputs/hf_archive.py" \
    --manifest "$NINFER_JOB_DIR/inputs/hf_archive_manifest.json" --output "$NINFER_JOB_DIR"
