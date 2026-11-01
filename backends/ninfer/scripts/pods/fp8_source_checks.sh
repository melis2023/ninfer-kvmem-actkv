#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
bash scripts/pods/converter_checks.sh
"$root/py311/bin/python" tests/convert/test_block_fp8.py \
    "$root/jobs/fp8-source-probe/subset"
