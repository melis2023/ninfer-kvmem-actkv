#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_XET_CACHE=/workspace/ninfer-work/ngram-corpus/xet
/root/.local/bin/uv pip install --python /workspace/ninfer-work/py311/bin/python pyarrow tokenizers
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/ngram_corpus_scan.py"
