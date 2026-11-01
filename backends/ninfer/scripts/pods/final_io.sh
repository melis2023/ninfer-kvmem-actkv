#!/usr/bin/env bash
set -Eeuo pipefail
bash scripts/pods/build.sh --target ninfer_tests ninfer-serve
ctest --test-dir /workspace/ninfer-work/build -R '^ninfer_load_report_test$' --output-on-failure
/workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/final_serving.py" --io-only
