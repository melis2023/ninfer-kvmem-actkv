#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
# ngram_cpu_checks.sh qualifies the row reader independently of this CUDA bundle.
ctest --test-dir "$root/build" --output-on-failure -V -R '^ninfer_cli_options_test$'
export NINFER_FLASH_NEXT_NGRAM_ONLY=1
bash scripts/pods/native_engine_checks.sh > "$NINFER_JOB_DIR/engine.log" 2>&1
tail -n 1 "$NINFER_JOB_DIR/engine.log"
bash scripts/pods/ngram_compare.sh
echo NGRAM_ROW_CACHE_CHECKS_PASS
