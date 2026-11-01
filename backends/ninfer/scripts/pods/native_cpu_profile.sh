#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH=/workspace/ninfer-work/src
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/native_cpu_profile.py"
