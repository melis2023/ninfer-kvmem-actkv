#!/usr/bin/env bash
set -Eeuo pipefail
export CUDA_MODULE_LOADING=LAZY
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/ngram_profile_engine.py"
