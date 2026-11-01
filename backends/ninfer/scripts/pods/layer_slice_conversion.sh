#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
"$root/py311/bin/python" -m scripts.pods.layer_slice
