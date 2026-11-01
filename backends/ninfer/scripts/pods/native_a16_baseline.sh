#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
baseline="$root/baselines/native-a16-vector-20261008"
mkdir -p "$baseline"
test ! -e "$baseline/ninfer_benches"
cp -p "$root/build/bench/ninfer_benches" "$baseline/ninfer_benches"
sha256sum "$baseline/ninfer_benches" > "$NINFER_JOB_DIR/baseline.sha256"
