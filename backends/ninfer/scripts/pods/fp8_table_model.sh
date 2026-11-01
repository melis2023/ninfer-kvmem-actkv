#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
"$root/py311/bin/python" "$NINFER_JOB_DIR/inputs/fp8_table_model.py"
cp "$root/sources/flash-next-bf16-ngram/fetch-manifest.json" "$NINFER_JOB_DIR/"
