#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH=/workspace/ninfer-work/src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
/workspace/ninfer-work/py311/bin/python -m scripts.pods.layer_slice --layers 1..4
