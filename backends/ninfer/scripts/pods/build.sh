#!/usr/bin/env bash
set -Eeuo pipefail
build=${NINFER_BUILD_DIR:-/workspace/ninfer-work/build}
arch=${NINFER_CUDA_ARCH:-86}
# nvcc's concurrent temporary files can fill this rental's model volume before linking. Keep
# them in the host's tmpfs; the build's memory-derived worker limit still applies.
export TMPDIR
TMPDIR=$(mktemp -d /dev/shm/ninfer-build-XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT
ulimit -c 0
# The size-focused compressor occupied individual CPU cores for minutes in this build. Balance
# changes fatbinary packaging only; retain completed objects and use it for subsequent compiles.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} --compress-mode=balance"
echo 'nvcc fatbinary compression: balance'
export NINFER_BUILD_ID=$(/workspace/ninfer-work/py311/bin/python -c \
    'import json; d=json.load(open("/workspace/ninfer-work/source.json")); print(d["commit"][:12]+"-snapshot-"+d["source_sha256"][:12])')
# A compiler cache carried between rentals (ccache_sync.py): a pod whose sources match an earlier
# rental's compiles only what changed. Every pod builds in /workspace/ninfer-work, which the entries
# need: most compile lines carry -ffile-prefix-map=<sources>=., which ccache's base directory does
# not rewrite, so a tree at another path misses most entries. ccache hashes each command
# line, and nvcc reads NVCC_APPEND_FLAGS from the environment, which ccache does not see: keep it
# the same for every build that shares this cache.
launcher=()
if command -v ccache >/dev/null; then
    export CCACHE_DIR=${CCACHE_DIR:-/workspace/ccache} CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-40G}
    export CCACHE_COMPILERCHECK=content CCACHE_BASEDIR=${CCACHE_BASEDIR:-$(dirname "$PWD")}
    [ -d "$CCACHE_DIR" ] || /workspace/ninfer-work/py311/bin/python scripts/pods/ccache_sync.py pull \
        --arch "$arch" --dir "$CCACHE_DIR"
    ccache --zero-stats >/dev/null
    launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
              -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
fi
if [ ! -f "$build/CMakeCache.txt" ] || [ ! -f "$build/build.ninja" ] || \
   [ ! -f "$build/src/runtime/engine/device_profiles_builtin.cpp" ]; then
    cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$arch" -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON \
        -DNINFER_BUILD_BENCHMARKS=ON -DPython3_EXECUTABLE=/workspace/ninfer-work/py311/bin/python \
        ${launcher[@]+"${launcher[@]}"}
fi
# Each nvcc can use several GiB; the container's cgroup (v2 or v1) RAM and CPU quota, not the
# host's totals, bound the parallelism.
jobs=$(/workspace/ninfer-work/py311/bin/python - <<'PY'
import math, os
from pathlib import Path

def first(*paths):
    for path in paths:
        file = Path(path)
        if file.exists():
            return file.read_text().split()
    return []

ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
limit = first("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
if limit and limit[0].isdigit() and int(limit[0]) < ram:
    ram = int(limit[0])
cpus = os.cpu_count()
quota = first("/sys/fs/cgroup/cpu.max")
if quota and quota[0].isdigit():
    cpus = min(cpus, math.ceil(int(quota[0]) / int(quota[1])))
else:
    v1 = first("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") + first("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if len(v1) == 2 and v1[0].isdigit():
        cpus = min(cpus, math.ceil(int(v1[0]) / int(v1[1])))
print(max(1, min(cpus, ram // (3 * 1024**3))))
PY
)
echo "build jobs: $jobs"
started=$SECONDS
cmake --build "$build" -j "$jobs" "$@"
echo "build seconds: $((SECONDS - started))"
if command -v ccache >/dev/null; then
    ccache --show-stats | grep -E "Hits|Misses|Cache size" || true
    [ "${NINFER_CCACHE_PUSH:-1}" = 0 ] || \
        /workspace/ninfer-work/py311/bin/python scripts/pods/ccache_sync.py push --arch "$arch" \
            --dir "$CCACHE_DIR"
fi
sha256sum "$build/apps/ninfer" "$build/apps/ninfer-serve" > ../jobs/build-binaries.sha256
