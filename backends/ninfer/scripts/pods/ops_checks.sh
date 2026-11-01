#!/usr/bin/env bash
set -Eeuo pipefail
ctest --test-dir /workspace/ninfer-work/build --output-on-failure -j2 \
    -R 'ninfer_(mtp_(pack|round)|speculative_mtp_(onehot|distribution)|qsa_indexer|sparse_softmax_attention|ple_inject|gdn_replay_fold|moe_experts_gguf|gguf_linear)_test$'
