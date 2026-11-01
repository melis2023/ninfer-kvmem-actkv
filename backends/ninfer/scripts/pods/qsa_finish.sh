#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
python="$root/py311/bin/python"
if ! "$python" -c 'import pytest' 2>/dev/null; then
    /root/.local/bin/uv pip install --python "$python" 'pytest==9.1.1'
fi
ctest --test-dir "$root/build" -R '^ninfer_qwen4_exp_ngram_writer_interop_test$' --output-on-failure
"$root/build/bench/ninfer_benches" ninfer_qsa_indexer_bench
