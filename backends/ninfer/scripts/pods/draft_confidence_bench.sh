#!/usr/bin/env bash
set -Eeuo pipefail
export NINFER_SERVE_BINARY=/workspace/ninfer-work/confidence-runtime/apps/ninfer-serve
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/final_serving.py" --confidence-only
