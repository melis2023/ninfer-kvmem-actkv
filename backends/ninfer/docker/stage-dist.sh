#!/usr/bin/env bash
# Stages what the container image needs from one NInfer build tree configured with
# NINFER_MULTICALL=ON (scripts/build-native.sh): the multi-call executable, stripped, as
# <dist>/bin/sm<arch>/ninfer-multicall, and in <dist>/packages/sm<arch>.txt the Debian packages that
# provide its shared libraries -- FFmpeg, libcurl and the CUDA libraries it links (cuBLAS, the CUDA
# runtime) -- so the runtime stage installs exactly those. The driver library comes from the host.
# Used by the Dockerfile's build stage and by CI, whose build jobs hand the same layout to the image.
#
#   docker/stage-dist.sh <build dir> <arch> <dist dir>
set -euo pipefail
(( $# == 3 )) || { printf 'usage: stage-dist.sh <build dir> <arch> <dist dir>\n' >&2; exit 2; }
build_dir="$1" arch="$2" dist="$3"
bin="$dist/bin/sm$arch" packages="$dist/packages/sm$arch.txt"

mkdir -p -- "$bin" "$dist/packages"
# The host symbol table is not needed at run time; the fatbins and the build-id note stay.
install -m 0755 -- "$build_dir/apps/ninfer-multicall" "$bin/ninfer-multicall"
strip --strip-unneeded -- "$bin/ninfer-multicall"
# ldd resolves through the build's own library path, dpkg -S names the owning package.
ldd "$bin/ninfer-multicall" | awk '$2 == "=>" && $3 ~ /^\// { print $3 }' | sort -u \
  | while read -r library; do
      dpkg -S "$(readlink -f -- "$library")" 2>/dev/null || dpkg -S "$library" 2>/dev/null || true
    done \
  | cut -d: -f1 | sort -u > "$packages"
printf 'staged %s (%s bytes): %s\n' "$bin/ninfer-multicall" "$(stat -c %s "$bin/ninfer-multicall")" \
  "$(tr '\n' ' ' < "$packages")"
