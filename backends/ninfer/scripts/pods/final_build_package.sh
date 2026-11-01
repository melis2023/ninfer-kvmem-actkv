#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
bash scripts/pods/build.sh --target ninfer_tests ninfer-perplexity
cd "$root/build"
sha256sum tests/ninfer_tests apps/ninfer-perplexity > "$NINFER_JOB_DIR/runtime.sha256"
ldd apps/ninfer-perplexity > "$NINFER_JOB_DIR/libraries.txt"
tar -I 'gzip -1' -cf "$NINFER_JOB_DIR/runtime.tar.gz" \
    tests/ninfer_tests apps/ninfer-perplexity \
    -C "$NINFER_JOB_DIR" runtime.sha256 source.json
stat -c '%n %s bytes' "$NINFER_JOB_DIR/runtime.tar.gz"
