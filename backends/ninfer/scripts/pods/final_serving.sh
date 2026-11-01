#!/usr/bin/env bash
set -Eeuo pipefail
bash scripts/pods/build.sh --target ninfer-serve
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/final_serving.py"
