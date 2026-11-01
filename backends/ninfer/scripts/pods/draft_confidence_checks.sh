#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
# The retained runtime volume has 17 GiB free; keep compiler objects in its 62 GiB tmpfs.
export NINFER_BUILD_DIR=/dev/shm/ninfer-confidence-build
bash "$NINFER_JOB_DIR/inputs/build.sh" --target ninfer_tests ninfer-serve ninfer
# /dev/shm is mounted noexec. Keep objects there and run the completed binaries from disk.
runtime="$root/confidence-runtime"
mkdir -p "$runtime/apps" "$runtime/tests"
cp "$NINFER_BUILD_DIR/apps/ninfer-serve" "$runtime/apps/"
cp "$NINFER_BUILD_DIR/tests/ninfer_tests" "$runtime/tests/"
export CUDA_MODULE_LOADING=LAZY
for test in ninfer_cli_options_test ninfer_serve_options_test ninfer_target_logprobs_test \
            ninfer_qwen4_exp_ngram_draft_prefetch_test; do
    timeout 60 "$runtime/tests/ninfer_tests" "$test"
done
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-native-q2-mtp.ninfer"
export NINFER_FLASH_NEXT_NGRAM_TABLE="$root/models/flash-next-iq4-table.ninfer"
export NINFER_FLASH_NEXT_CONFIDENCE_ONLY=1
export NINFER_FLASH_NEXT_CONFIDENCE_BATCHED=1
timeout 300 "$runtime/tests/ninfer_tests" ninfer_qwen4_exp_engine_real
echo DRAFT_CONFIDENCE_CHECKS_PASS
