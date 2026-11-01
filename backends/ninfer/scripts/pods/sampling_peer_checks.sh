#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-q2_0-mtp.ninfer"
ctest --test-dir "$root/build" --output-on-failure -V \
    -R '^ninfer_qwen4_exp_peer_copy_order_test$'
