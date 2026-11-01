#!/usr/bin/env bash
# The typed decision endpoints under the SystemOne bench of issue #64 (bench.mjs and data.mjs as
# job inputs): decision_ab.py over the given configurations. Node.js comes from nodejs.org when the
# image has none.
set -Eeuo pipefail
root=/workspace/ninfer-work
if ! command -v node >/dev/null; then
    version=v22.11.0
    [ -x "$root/node/bin/node" ] || {
        mkdir -p "$root/node"
        curl -fsSL "https://nodejs.org/dist/$version/node-$version-linux-x64.tar.xz" |
            tar -xJ -C "$root/node" --strip-components=1
    }
    export PATH="$root/node/bin:$PATH"
fi
node --version
while [ ! -f "$root/jobs/models/exit" ]; do sleep 2; done
[ "$(cat "$root/jobs/models/exit")" = 0 ] || exit 125
"$root/py311/bin/python" -u scripts/pods/decision_ab.py --bench "$NINFER_JOB_DIR/inputs" \
    --out "$NINFER_JOB_DIR/ab" "$@"
