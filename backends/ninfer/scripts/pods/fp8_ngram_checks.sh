#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
bash scripts/pods/converter_checks.sh
ctest --test-dir "$root/build" --output-on-failure \
    -R '^(ninfer_artifact_reader_test|ninfer_ngram_rows_test|ninfer_qwen4_exp_ngram_(hash|table|component|writer_interop)_test)$'
