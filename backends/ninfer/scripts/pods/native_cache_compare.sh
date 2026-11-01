#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-native-q2-mtp.ninfer"
export NINFER_FLASH_NEXT_NGRAM_TABLE="$root/models/flash-next-q2_0-mtp.ninfer"
export NINFER_FLASH_NEXT_CACHE_ONLY=1
export NINFER_FLASH_NEXT_CACHE_COMPARE_ONLY=1
"$root/build/tests/ninfer_tests" ninfer_qwen4_exp_engine_real
