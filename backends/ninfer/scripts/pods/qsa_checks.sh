#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
ctest --test-dir "$root/build" -R '^ninfer_(qsa_indexer|qwen4_exp_ngram_table|qwen4_exp_ngram_component|qwen4_exp_ngram_writer_interop)_test$' --output-on-failure
"$root/build/bench/ninfer_benches" ninfer_qsa_indexer_bench
