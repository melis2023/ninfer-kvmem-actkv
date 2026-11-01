#!/usr/bin/env bash
set -Eeuo pipefail
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/final_serving.py" --plain-only
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/final_serving.py" --gguf-baseline
