#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
"$root/py311/bin/python" -m pytest -q \
    tests/reference/test_fetch_slice.py tests/convert/test_official_recipes.py \
    tests/convert/test_qwen4_exp.py tests/convert/test_qwen4_exp_gguf.py \
    tests/convert/test_recipe.py tests/convert/test_q2_native_import.py \
    tests/artifact/test_fp8_row.py tests/convert/quantization/test_fp8_row.py \
    tests/convert/test_qwen4_exp_ngram.py tests/convert/test_block_fp8.py \
    tests/convert/test_sources.py tests/convert/test_qwen3_5.py
