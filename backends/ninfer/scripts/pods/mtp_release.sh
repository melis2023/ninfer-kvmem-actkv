#!/usr/bin/env bash
set -Eeuo pipefail
export HF_TOKEN_PATH=/run/ninfer-hf/token
trap 'rm -f -- /run/ninfer-hf/token' EXIT
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE=/workspace/ninfer-work/mtp-release/xet
/workspace/ninfer-work/py311/bin/python -m scripts.pods.mtp_release \
    --audit "$NINFER_JOB_DIR/inputs/audit.json" \
    --root /workspace/ninfer-work/mtp-release \
    --receipts "$NINFER_JOB_DIR/receipts"
