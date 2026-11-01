#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
test "$(cat "$root/jobs/layer-slice-ple-convert/exit")" = 0
test "$(cat "$root/jobs/standalone-table/exit")" = 0
# Conversion evidence remains valid across Engine-only source updates.
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-native-q2-layers-1-4.ninfer"
export NINFER_FLASH_NEXT_NGRAM_TABLE="$root/models/flash-next-iq4-table.ninfer"
export NINFER_FLASH_NEXT_SLICE_ONLY=1
"$root/build/tests/ninfer_tests" ninfer_qwen4_exp_engine_real
