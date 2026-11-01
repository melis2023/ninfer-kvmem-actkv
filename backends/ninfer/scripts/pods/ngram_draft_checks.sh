#!/usr/bin/env bash
set -Eeuo pipefail
ctest --test-dir /workspace/ninfer-work/build --output-on-failure -V \
    -R '^ninfer_(argmax|qwen4_exp_ngram_draft_prefetch)_test$'
