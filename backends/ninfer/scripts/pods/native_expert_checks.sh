#!/usr/bin/env bash
set -Eeuo pipefail
ctest --test-dir /workspace/ninfer-work/build --output-on-failure -V \
    -R '^ninfer_(moe_experts_native|moe_expert_cpu|qwen4_exp_hybrid_experts|qwen4_exp_expert_profile|artifact_materialization|residual_add)_test$'
