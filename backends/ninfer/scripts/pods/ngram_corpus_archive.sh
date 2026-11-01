#!/usr/bin/env bash
set -Eeuo pipefail
export HF_TOKEN_PATH=/run/ninfer-hf/token
trap 'rm -f -- /run/ninfer-hf/token' EXIT
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/ngram_corpus_archive.py"
