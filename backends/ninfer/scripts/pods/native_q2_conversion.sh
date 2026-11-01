#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1
/workspace/ninfer-work/py311/bin/python -m scripts.pods.convert_native_q2 --refresh
