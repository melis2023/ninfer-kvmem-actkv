#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH=/workspace/ninfer-work/src
/workspace/ninfer-work/py311/bin/python -m scripts.pods.pipeline_run --devices 0,1
