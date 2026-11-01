#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
ctest --test-dir "$root/build" -R '^ninfer_qsa_indexer_test$' --output-on-failure
"$root/build/bench/ninfer_benches" ninfer_qsa_indexer_bench
