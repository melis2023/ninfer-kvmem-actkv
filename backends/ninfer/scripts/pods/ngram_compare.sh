#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
# One comparison only; the Engine checks cover cache-off and direct I/O separately. Normal
# startup/warmup remains in both runs, after mincore-verified eviction of only the table's pages.
prompt='Read the following repeated list and write a Python function to merge two sorted lists. Include code only.'
for ((i=0; i<150; ++i)); do prompt+=' alpha beta gamma delta'; done
common=("$root/models/flash-next-native-q2-mtp.ninfer" --devices 0,1 --greedy --no-thinking \
    --ngram-table "$root/models/flash-next-q2_0-mtp.ninfer" --max-new 48 \
    --max-context 4096 --kv-capacity 4096 --kv-dtype int8 --spec mtp --draft-tokens 4 \
    --ngram-draft-tokens 0 --prompt "$prompt")
for mode in baseline cache; do
    binary="$root/build/apps/ninfer"
    extra=()
    if [ "$mode" = baseline ]; then
        binary="$root/baselines/ngram-pre-cache-20261008/ninfer"
    else
        extra=(--ngram-ram-mib 4096)
    fi
    tag="$mode-1"
    "$root/py311/bin/python" scripts/pods/ngram_cache.py \
        "$root/models/flash-next-q2_0-mtp.ninfer" --discard > "$NINFER_JOB_DIR/$tag-pages.json"
    "$binary" "${common[@]}" "${extra[@]}" \
        > "$NINFER_JOB_DIR/$tag.txt" 2> "$NINFER_JOB_DIR/$tag.err"
    cmp "$NINFER_JOB_DIR/baseline-1.txt" "$NINFER_JOB_DIR/$tag.txt"
    echo "NGRAM_MEASURED $tag"
done
echo NGRAM_COMPARISON_PASS
