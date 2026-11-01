#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
stage="$NINFER_JOB_DIR/runtime"
package="$NINFER_JOB_DIR/inputs/runtime.tar.gz"
if [ ! -f "$package" ]; then
    package="$root/jobs/runtime-package/runtime.tar.gz"
    for attempt in $(seq 1 60); do
        if [ -f "$root/jobs/runtime-package/exit" ] && [ -f "$package" ]; then break; fi
        sleep 5
    done
    if [ ! -f "$package" ]; then
        echo 'Runtime package did not arrive within 300 seconds.' >&2
        exit 1
    fi
fi
mkdir "$stage"
tar xf "$package" -C "$stage"
cmp -s "$stage/source.json" "$root/source.json"
cd "$stage"
sha256sum -c runtime.sha256
ldd tests/ninfer_tests > "$NINFER_JOB_DIR/libraries.txt"
if grep -q 'not found' "$NINFER_JOB_DIR/libraries.txt"; then
    cat "$NINFER_JOB_DIR/libraries.txt"
    exit 1
fi
mkdir -p "$root/build/tests"
if [ -e "$root/build/tests/ninfer_tests" ]; then
    cmp -s tests/ninfer_tests "$root/build/tests/ninfer_tests"
else
    cp tests/ninfer_tests "$root/build/tests/ninfer_tests"
fi
echo RUNTIME_INSTALL_PASS
