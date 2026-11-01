#!/usr/bin/env bash
set -Eeuo pipefail
export HF_TOKEN_PATH=/run/ninfer-hf/token.profile.publish
trap 'rm -f -- "$HF_TOKEN_PATH"' EXIT
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/ngram_profile_publish.py"
