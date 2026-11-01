#!/usr/bin/env bash
set -Eeuo pipefail
export NINFER_QWEN4_EXP_ARTIFACT=/workspace/ninfer-work/models/flash-next-q2_0-mtp.ninfer
export NINFER_QWEN4_EXP_DEVICES=0,1
export NINFER_QWEN4_EXP_DRAFTS=4
/workspace/ninfer-work/build/tests/ninfer_tests ninfer_qwen4_exp_generate_real
