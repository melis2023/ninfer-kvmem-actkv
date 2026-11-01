#!/usr/bin/env bash
# Container entrypoint. Picks the build that matches the GPUs -- sm75 for compute capability 7.5,
# sm86 for 8.6 and 8.9 (sm_86 cubins run on Ada), sm120a for 12.0 -- and runs one of:
#
#   run <model> [profile]     scripts/run.sh: the measured serving profiles (default: run qwen38-27b)
#   download <model>          scripts/download-model.sh into /models
#   ninfer | serve | perplexity | calibrate | ngram-profile [args]   the binary itself; serve
#                             listens on NINFER_HOST:NINFER_PORT unless --host/--port are given
#   anything else             executed as given
#
# GPUs that need different builds (a layer pipeline with --devices across, say, an RTX 2080 Ti and
# an RTX 3090) take `multi`, a build containing all selected architectures,
# where every card runs its own kernels. Without it they take the build of the oldest
# card, which the newer cards run from its PTX: slower there, and compiled at the first start.
# An image that carries only `multi` runs it on every GPU.
#
# NINFER_IMAGE_ARCH=sm75|sm86|sm120a|multi skips detection.
set -euo pipefail
home=/opt/ninfer
: "${NINFER_HOST:=0.0.0.0}" "${NINFER_PORT:=8080}"

# The build of one compute capability, or nothing for an unsupported one.
build_of() {
  case "${1// /}" in
    7.5) printf 'sm75' ;;
    8.6|8.7|8.9) printf 'sm86' ;;
    12.0) printf 'sm120a' ;;
  esac
}

select_arch() {
  if [[ -n "${NINFER_IMAGE_ARCH:-}" ]]; then
    printf '%s' "$NINFER_IMAGE_ARCH"
    return
  fi
  local capabilities
  if ! capabilities="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" ||
     [[ -z "$capabilities" ]]; then
    printf 'No GPU visible in the container. Start it with --gpus all (docker run) or a GPU\n' >&2
    printf 'reservation (compose), with the NVIDIA Container Toolkit installed on the host.\n' >&2
    exit 1
  fi
  local builds=() capability build
  while read -r capability; do
    [[ -n "$capability" ]] || continue
    build="$(build_of "$capability")"
    if [[ -z "$build" ]]; then
      printf 'Compute capability %s is not supported: NInfer runs on 7.5 (RTX 20, TITAN RTX), 8.6\n' "$capability" >&2
      printf '(RTX 30), 8.9 (RTX 40) and 12.0 (RTX 50). NINFER_IMAGE_ARCH=sm75|sm86|sm120a|multi\n' >&2
      printf 'overrides the detection; CUDA_VISIBLE_DEVICES hides a GPU from the container.\n' >&2
      exit 1
    fi
    builds+=("$build")
  done <<<"$capabilities"
  local unique
  unique="$(printf '%s\n' "${builds[@]}" | sort -u)"
  if [[ -x "$home/multi/ninfer-serve" && ( "$(wc -l <<<"$unique")" -gt 1 || ! -d "$home/${builds[0]}" ) ]]; then
    printf 'multi'
  elif [[ "$(wc -l <<<"$unique")" -gt 1 ]]; then
    # The oldest card's build also runs on the newer ones.
    for build in sm75 sm86 sm120a; do
      grep -qxF "$build" <<<"$unique" && break
    done
    printf 'GPUs of several architectures (%s): running the %s build on all of them; a multiarch\n' \
      "$(tr '\n' ' ' <<<"$unique" | sed 's/ $//')" "$build" >&2
    printf 'image runs each card on its own code.\n' >&2
    printf '%s' "$build"
  else
    printf '%s' "${builds[0]}"
  fi
}

command="${1:-run}"
(( $# )) && shift
case "$command" in
  download)
    exec "$home/download-model.sh" "$@" ;;
esac

arch="$(select_arch)"
bin="$home/$arch"
[[ -x "$bin/ninfer-serve" ]] || { printf 'This image has no %s build.\n' "$arch" >&2; exit 1; }

case "$command" in
  run)
    model="${1:-qwen38-27b}"
    graft_dir="${NINFER_GRAFT_DIR:-/grafts/$model}"
    exec env NINFER_SERVER="$bin/ninfer-serve" NINFER_GRAFT_DIR="$graft_dir" \
      "$home/run.sh" "$model" "${@:2}" ;;
  ninfer) exec "$bin/ninfer" "$@" ;;
  serve)
    # ninfer-serve binds 127.0.0.1 by default, which nothing outside the container reaches.
    listen=()
    [[ " $* " == *" --host "* ]] || listen+=(--host "$NINFER_HOST")
    [[ " $* " == *" --port "* ]] || listen+=(--port "$NINFER_PORT")
    exec "$bin/ninfer-serve" "$@" ${listen[@]+"${listen[@]}"} ;;
  perplexity) exec "$bin/ninfer-perplexity" "$@" ;;
  calibrate) exec "$bin/ninfer-calibrate" "$@" ;;
  ngram-profile) exec "$bin/ninfer-ngram-profile" "$@" ;;
  *) exec "$command" "$@" ;;
esac
