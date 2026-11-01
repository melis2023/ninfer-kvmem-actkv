#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
corpus=eval/corpora/perplexity-heldout-2026-09/manifest.json
for table in iq4 fp8; do
    artifact="$root/models/flash-next-native-q2-mtp.ninfer"
    if [ "$table" = fp8 ]; then artifact="$root/models/flash-next-native-q2-mtp-fp8-table.ninfer"; fi
    "$root/build/apps/ninfer-perplexity" "$artifact" \
        --devices 0,1 --ngram-table "$root/models/flash-next-$table-table.ninfer" \
        --corpus "$corpus" --quick --context 32768 --stride 16384 --kv-dtype bf16 \
        --output "$NINFER_JOB_DIR/$table"
done
echo FP8_IQ4_QUALITY_READY
