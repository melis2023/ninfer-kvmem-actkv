#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
# Dispatch after build when the artifact is ready, or after native-convert when it is being made.
# Never wait for a CPU prerequisite while holding the GPU lock.
for prerequisite in build native-convert; do
    [ -f "$root/jobs/$prerequisite/exit" ] || exit 125
    [ "$(cat "$root/jobs/$prerequisite/exit")" = 0 ] || exit 125
done
cmp -s "$NINFER_JOB_DIR/source.json" "$root/jobs/build/source.json" || exit 125
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-native-q2-mtp.ninfer"
export NINFER_FLASH_NEXT_NGRAM_TABLE="$root/models/flash-next-q2_0-mtp.ninfer"
"$root/build/tests/ninfer_tests" ninfer_qwen4_exp_engine_real
