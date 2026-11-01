#!/usr/bin/env bash
# Keep release and CI compilation inputs identical; CI additionally enables test targets.
set -euo pipefail
: "${ARCHS:?}"
case "${1:-OFF}" in ON|OFF) testing="${1:-OFF}" ;; *) exit 2 ;; esac
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DNINFER_SHARED_RUNTIME=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DBUILD_TESTING="$testing" -DNINFER_BUILD_BENCHMARKS=ON
