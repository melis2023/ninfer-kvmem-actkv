#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_XET_CACHE=/workspace/ninfer-work/ngram-corpus/xet
export PYTHONPATH=/workspace/ninfer-work/src
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/ngram_corpus_build.py" \
    --audit "$NINFER_JOB_DIR/inputs/audit.json"
