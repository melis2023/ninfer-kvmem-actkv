#!/usr/bin/env bash
set -Eeuo pipefail
bash scripts/pods/build.sh --target ninfer ninfer_tests ninfer-serve ninfer_artifact_materialization_test
