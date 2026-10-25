#!/usr/bin/env bash
# Native Linux build in explicit steps, the counterpart of build-native.bat: CUDA from
# NINFER_CUDA_PATH, FFmpeg and libcurl from pkg-config, optionally under a prebuilt dependency
# prefix (NINFER_PREFIX, added to CMAKE_PREFIX_PATH and PKG_CONFIG_PATH). The Docker image and CI
# build through this script. Every setting below can be overridden from the environment.
#
#   scripts/build-native.sh configure
#   scripts/build-native.sh build
#   scripts/build-native.sh target ninfer-serve [more targets]
#   scripts/build-native.sh test        ctest; a test that needs a GPU skips without one
#
#   NINFER_CUDA_PATH   CUDA toolkit root            (default /usr/local/cuda)
#   NINFER_CUDA_ARCH   80, 86, 89 or 120a           (default 86)
#   NINFER_BUILD_DIR   build tree                   (default build-native)
#   NINFER_TESTS       ON or OFF: build the tests   (default ON)
#   NINFER_MULTICALL   ON or OFF: the programs as one multi-call executable, ninfer-multicall,
#                      with their names as symlinks (apps/CMakeLists.txt)   (default OFF)
#   NINFER_PREFIX      prebuilt dependency prefix   (default none)
#   NINFER_JOBS        build parallelism            (default: all cores)
set -euo pipefail

cuda_path="${NINFER_CUDA_PATH:-/usr/local/cuda}"
arch="${NINFER_CUDA_ARCH:-86}"
build_dir="${NINFER_BUILD_DIR:-build-native}"
tests="${NINFER_TESTS:-ON}"
multicall="${NINFER_MULTICALL:-OFF}"
jobs="${NINFER_JOBS:-$(nproc)}"

case "$arch" in 80|86|89|120a) ;; *) printf 'NINFER_CUDA_ARCH must be 80, 86, 89 or 120a, got %s\n' "$arch" >&2; exit 2 ;; esac
case "$tests" in ON|OFF) ;; *) printf 'NINFER_TESTS must be ON or OFF, got %s\n' "$tests" >&2; exit 2 ;; esac
case "$multicall" in ON|OFF) ;; *) printf 'NINFER_MULTICALL must be ON or OFF, got %s\n' "$multicall" >&2; exit 2 ;; esac

export CUDACXX="$cuda_path/bin/nvcc"
export PATH="$cuda_path/bin:$PATH"
prefix_args=()
if [[ -n "${NINFER_PREFIX:-}" ]]; then
  export PKG_CONFIG_PATH="$NINFER_PREFIX/lib/pkgconfig:$NINFER_PREFIX/lib64/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  prefix_args=(-DCMAKE_PREFIX_PATH="$NINFER_PREFIX")
fi
launcher_args=()
if command -v ccache >/dev/null; then
  launcher_args=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
                 -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

case "${1:-}" in
  configure)
    cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="$arch" -DBUILD_TESTING="$tests" -DNINFER_MULTICALL="$multicall" \
      ${prefix_args[@]+"${prefix_args[@]}"} ${launcher_args[@]+"${launcher_args[@]}"} ;;
  build)
    cmake --build "$build_dir" --parallel "$jobs" ;;
  target)
    shift
    (( $# )) || { printf 'usage: build-native.sh target <target> [<target> ...]\n' >&2; exit 2; }
    cmake --build "$build_dir" --parallel "$jobs" --target "$@" ;;
  test)
    ctest --test-dir "$build_dir" --output-on-failure -j "$jobs" ;;
  *)
    printf 'usage: build-native.sh configure|build|target <target>...|test\n' >&2
    exit 2 ;;
esac
