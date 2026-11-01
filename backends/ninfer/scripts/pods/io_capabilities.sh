#!/usr/bin/env bash
set -Eeuo pipefail
g++ -std=c++20 -O2 "$NINFER_JOB_DIR/inputs/io_uring_probe.cpp" \
    -o "$NINFER_JOB_DIR/io-uring-probe"
"$NINFER_JOB_DIR/io-uring-probe"
