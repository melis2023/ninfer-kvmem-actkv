#!/usr/bin/env bash
set -Eeuo pipefail
export HF_TOKEN_PATH=/run/ninfer-hf/token.profile
trap 'rm -f -- /run/ninfer-hf/token.profile' EXIT
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
/root/.local/bin/uv pip install --python /workspace/ninfer-work/py311/bin/python 'tokenizers==0.23.2'
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/ngram_profile_broad.py"
