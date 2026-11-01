#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
mkdir "$NINFER_JOB_DIR/runtime"
cd "$NINFER_JOB_DIR/runtime"
tar xf "$NINFER_JOB_DIR/inputs/runtime.tar.gz"
cmp -s source.json "$root/source.json"
sha256sum -c runtime.sha256
ldd tests/ninfer_tests apps/ninfer-perplexity > "$NINFER_JOB_DIR/libraries.txt"
if grep -q 'not found' "$NINFER_JOB_DIR/libraries.txt"; then
    cat "$NINFER_JOB_DIR/libraries.txt"
    exit 1
fi
mkdir -p "$root/build/tests" "$root/build/apps"
for binary in tests/ninfer_tests apps/ninfer-perplexity; do
    if [ -e "$root/build/$binary" ]; then
        mv "$root/build/$binary" "$NINFER_JOB_DIR/previous-${binary##*/}"
    fi
    mv "$binary" "$root/build/$binary"
done
echo FINAL_RUNTIME_INSTALLED
