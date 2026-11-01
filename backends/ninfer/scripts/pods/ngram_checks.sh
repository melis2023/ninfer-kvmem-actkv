#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
build="$root/build"
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-q2_0-mtp.ninfer"
cache_helper="$root/src/scripts/pods/ngram_cache.py"
if [ -f "$NINFER_JOB_DIR/inputs/ngram_cache.py" ]; then
    cache_helper="$NINFER_JOB_DIR/inputs/ngram_cache.py"
fi
cat > "$NINFER_JOB_DIR/chat.jsonl" <<'JSONL'
{"messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],"enable_thinking":false}
{"messages":[{"role":"user","content":"Compute 2+2."}],"enable_thinking":true}
JSONL
"$build/apps/ninfer-ngram-profile" "$NINFER_FLASH_NEXT_ARTIFACT" \
    --out "$NINFER_JOB_DIR/chat.hot" "$NINFER_JOB_DIR/chat.jsonl" \
    > "$NINFER_JOB_DIR/profile-chat.txt"
"$build/apps/ninfer-ngram-profile" "$NINFER_FLASH_NEXT_ARTIFACT" \
    --out "$root/models/technical.hot" docs/qwen3-8-flash-next.md \
    docs/maintainer/engine-architecture.md > "$NINFER_JOB_DIR/profile-train.txt"
"$build/apps/ninfer-ngram-profile" "$NINFER_FLASH_NEXT_ARTIFACT" \
    --evaluate "$root/models/technical.hot" README.md docs/serving.md \
    > "$NINFER_JOB_DIR/profile-heldout.txt"
# Interleave modes over the same prompt. Preserve each run's output, timing and reader counters.
common=("$NINFER_FLASH_NEXT_ARTIFACT" --devices 0,1 --greedy --no-thinking --max-new 80 \
    --max-context 4096 --kv-capacity 4096 --kv-dtype int8 \
    --prompt 'Write a Python function that merges two sorted lists. Include code only.')
for round in cold warm; do
    for mode in buffered direct mmap ram ram-hot; do
        cache=("$root/py311/bin/python" "$cache_helper" "$NINFER_FLASH_NEXT_ARTIFACT")
        if [ "$round" = cold ]; then cache+=(--discard); fi
        "${cache[@]}" > "$NINFER_JOB_DIR/$mode-$round-cache.json"
        extra=(--ngram-io "$mode")
        if [ "$mode" = ram ]; then extra=(--ngram-residency ram); fi
        if [ "$mode" = ram-hot ]; then
            extra=(--ngram-residency ram-hot --ngram-hot-profile "$root/models/technical.hot" \
                --ngram-ram-mib 256)
        fi
        "$build/apps/ninfer" "${common[@]}" "${extra[@]}" \
            > "$NINFER_JOB_DIR/$mode-$round.txt" 2> "$NINFER_JOB_DIR/$mode-$round.err"
        cmp "$NINFER_JOB_DIR/buffered-cold.txt" "$NINFER_JOB_DIR/$mode-$round.txt"
    done
done
echo NGRAM_CHECKS_PASS
