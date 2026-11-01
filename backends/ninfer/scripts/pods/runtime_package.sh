#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
test "$(cat "$root/jobs/build/exit")" = 0
cmp -s "$root/jobs/build/source.json" "$root/source.json"
cd "$root/build"
sha256sum tests/ninfer_tests > "$NINFER_JOB_DIR/runtime.sha256"
ldd tests/ninfer_tests > "$NINFER_JOB_DIR/libraries.txt"
tar -I 'gzip -1' -cf "$NINFER_JOB_DIR/runtime.tar.gz" \
    -C "$root/build" tests/ninfer_tests \
    -C "$NINFER_JOB_DIR" runtime.sha256 source.json
stat -c '%n %s bytes' tests/ninfer_tests "$NINFER_JOB_DIR/runtime.tar.gz"
echo RUNTIME_PACKAGE_READY
