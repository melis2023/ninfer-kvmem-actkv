#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH=/workspace/ninfer-work/src
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export RUST_LOG=off
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/standalone_table.py"
