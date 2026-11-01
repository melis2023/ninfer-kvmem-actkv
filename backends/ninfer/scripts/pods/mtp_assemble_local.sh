#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE=/workspace/ninfer-work/mtp-release/xet
mkdir -p "$NINFER_JOB_DIR/python/scripts/pods"
cp "$NINFER_JOB_DIR/inputs/attach_mtp.py" "$NINFER_JOB_DIR/python/scripts/pods/attach_mtp.py"
export PYTHONPATH="$NINFER_JOB_DIR/python:/workspace/ninfer-work/src"
py=/workspace/ninfer-work/py311/bin/python
variant=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["variant"])' \
    "$NINFER_JOB_DIR/inputs/candidate.json")
case "$variant" in q2|iq3|coder) ;; *) exit 2 ;; esac
export HF_TOKEN_PATH="/run/ninfer-hf/token.assemble.$variant"
trap 'rm -f -- "$HF_TOKEN_PATH"' EXIT
"$py" "$NINFER_JOB_DIR/inputs/mtp_release.py" --local-only --variant "$variant" \
    --audit "$NINFER_JOB_DIR/inputs/audit.json" \
    --root /workspace/ninfer-work/mtp-release --receipts "$NINFER_JOB_DIR/receipts"
