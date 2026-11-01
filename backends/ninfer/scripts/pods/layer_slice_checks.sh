#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
test "$(cat "$root/jobs/layer-slice-convert/exit")" = 0
# Conversion evidence remains valid across Engine-only source updates.
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-native-q2-layers-3-5.ninfer"
export NINFER_FLASH_NEXT_SLICE_ONLY=1
unset NINFER_FLASH_NEXT_NGRAM_TABLE
ctest --test-dir "$root/build" --output-on-failure -R '^ninfer_qwen4_exp_config_test$'
"$root/build/tests/ninfer_tests" ninfer_qwen4_exp_engine_real
