#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
corpus=eval/corpora/perplexity-heldout-2026-09/manifest.json
for representation in gguf native; do
    case "$representation" in
        gguf) artifact="$root/models/flash-next-q2_0-mtp.ninfer" ;;
        native) artifact="$root/models/flash-next-native-q2-mtp.ninfer" ;;
    esac
    "$root/build/apps/ninfer-perplexity" "$artifact" \
        --ngram-table "$root/models/flash-next-iq4-table.ninfer" \
        --corpus "$corpus" --quick --context 32768 --stride 16384 --kv-dtype bf16 \
        --output "$NINFER_JOB_DIR/$representation"
done
echo NATIVE_QUALITY_REPORTS_READY
