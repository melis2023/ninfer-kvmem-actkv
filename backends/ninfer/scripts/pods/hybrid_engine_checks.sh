#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-native-q2-mtp.ninfer"
export NINFER_FLASH_NEXT_NGRAM_TABLE="$root/models/flash-next-iq4-table.ninfer"
export NINFER_FLASH_NEXT_HYBRID_ONLY=1
exec "$root/build/tests/ninfer_tests" ninfer_qwen4_exp_engine_real
