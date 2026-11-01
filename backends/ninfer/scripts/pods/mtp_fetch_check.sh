#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
"$root/py311/bin/python" -m tools.reference.fetch_slice \
    --repo Qwen/Qwen3.8-Flash-Next \
    --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
    --name mtp.pre_fc_norm_embedding.weight --out "$NINFER_JOB_DIR/subset"
sha256sum "$NINFER_JOB_DIR"/subset/*.safetensors > "$NINFER_JOB_DIR/subset-files.sha256"
