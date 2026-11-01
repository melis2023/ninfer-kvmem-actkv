#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
ctest --test-dir "$root/build" -R '^ninfer_qwen4_exp_ngram_draft_prefetch_test$' --output-on-failure
"$root/build/bench/ninfer_benches" ninfer_hyper_connection_bench
