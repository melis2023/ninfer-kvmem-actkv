#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
test=tests/artifact/test_q2_native_codec.py
if [ -f "$NINFER_JOB_DIR/inputs/test_q2_native_codec.py" ]; then
    test="$NINFER_JOB_DIR/inputs/test_q2_native_codec.py"
fi
PYTHONPATH="$root/src" "$root/py311/bin/python" -m pytest -q "$test" \
    tests/artifact/test_codecs.py tests/convert/test_q2_native_import.py \
    tests/convert/test_recipe.py tests/convert/test_gguf_blocks.py \
    tests/convert/test_qwen4_exp_gguf.py tests/convert/quantization/test_groupwise.py \
    tests/convert/quantization/test_groupwise_search.py
