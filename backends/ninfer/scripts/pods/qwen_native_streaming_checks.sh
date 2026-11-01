#!/usr/bin/env bash
set -Eeuo pipefail

# The controller selects the host and runs this payload in a dedicated checkout. No rental,
# artifact download or deployment is performed here.
build=${NINFER_NATIVE_BUILD:-build}
(
    while true; do
        printf 'native streaming check heartbeat %s\n' "$(date -u +%FT%TZ)"
        sleep 60
    done
) &
heartbeat=$!
trap 'kill "$heartbeat" 2>/dev/null || true' EXIT

"$build/tests/ninfer_tests" ninfer_qwen_native_expert_layout_test
"$build/tests/ninfer_tests" ninfer_sparse_moe_test --streaming
"$build/tests/ninfer_tests" ninfer_sparse_moe_test --wide-only
printf 'NATIVE_STREAMING_OP_CHECKS_PASS\n'
if [ -n "${NINFER_NATIVE_MOE_ARTIFACT:-}" ]; then
    "$build/tests/ninfer_tests" ninfer_qwen_native_streaming_real_test
    printf 'NATIVE_STREAMING_ENGINE_CHECKS_PASS\n'
fi
