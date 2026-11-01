#!/usr/bin/env bash
set -Eeuo pipefail
ctest --test-dir /workspace/ninfer-work/build --output-on-failure -j2 \
    -R '^ninfer_artifact_(reader|permute_row_split|writer_interop)_test$'
