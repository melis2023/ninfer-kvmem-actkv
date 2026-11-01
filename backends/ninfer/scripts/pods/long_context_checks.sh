#!/usr/bin/env bash
set -Eeuo pipefail
export NINFER_FLASH_NEXT_ARTIFACT=/workspace/ninfer-work/models/flash-next-q2_0-mtp.ninfer
/workspace/ninfer-work/build/tests/ninfer_tests ninfer_qwen4_exp_long_context_real
