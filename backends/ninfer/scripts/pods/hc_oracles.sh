#!/usr/bin/env bash
set -Eeuo pipefail
ctest --test-dir /workspace/ninfer-work/build \
    -R '^ninfer_(hyper_connection_test|qwen4_exp_ngram_(table|draft_prefetch)_test)$' \
    --output-on-failure
