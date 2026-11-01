#!/usr/bin/env bash
set -Eeuo pipefail
ctest --test-dir /workspace/ninfer-work/build --output-on-failure -V \
    -R '^ninfer_moe_experts_gguf_test$'
