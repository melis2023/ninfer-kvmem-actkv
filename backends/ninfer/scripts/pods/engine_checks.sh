#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
build="$root/build"
ctest --test-dir "$build" --output-on-failure -j2 \
    -R 'qwen4_exp_(ngram_hash|ngram_table|ngram_component|config|expert_cache|expert_stream)|prompt_input|speculative_accept|qsa|ple_|hyper_connection'
while [ ! -f "$root/jobs/models/exit" ]; do sleep 2; done
[ "$(cat "$root/jobs/models/exit")" = 0 ] || exit 125
# Runtime-only source changes reuse the artifact prepared from pinned model inputs.
# The Engine loader validates its framing, component bindings and companion table digest.
export NINFER_FLASH_NEXT_ARTIFACT="$root/models/flash-next-q2_0-mtp.ninfer"
set +e
"$build/tests/ninfer_tests" ninfer_qwen4_exp_engine_real
engine_rc=$?
export NINFER_LOGPROBS_ARTIFACT="$NINFER_FLASH_NEXT_ARTIFACT"
export NINFER_LOGPROBS_DEVICES=0,1
export NINFER_LOGPROBS_SPECULATIVE=mtp
"$build/tests/ninfer_tests" ninfer_engine_logprobs_real_test
logprobs_rc=$?
set -e
if [ "$engine_rc" != 0 ] || [ "$logprobs_rc" != 0 ]; then exit 1; fi
echo ENGINE_CHECKS_PASS
