#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
"$root/build/apps/ninfer" "$root/models/flash-next-q2_0-mtp.ninfer" \
    --devices 0,1 --expert-residency host --expert-cache-mib 256 \
    --greedy --no-thinking --max-new 80 --max-context 4096 --kv-capacity 4096 --kv-dtype int8 \
    --prompt 'Write a Python function that merges two sorted lists. Include code only.' \
    > "$NINFER_JOB_DIR/answer.txt" 2> "$NINFER_JOB_DIR/metrics.txt"
cmp "$root/jobs/ngram-checks/buffered-cold.txt" "$NINFER_JOB_DIR/answer.txt"
echo HOST_CACHE_CHECK_PASS
