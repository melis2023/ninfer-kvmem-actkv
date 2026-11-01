#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH="$NINFER_JOB_DIR/inputs:/workspace/ninfer-work/src"
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/ngram_profile_attach.py"
