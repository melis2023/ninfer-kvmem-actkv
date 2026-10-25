#!/usr/bin/env bash
# Container entrypoint. Picks the build that matches the GPU -- sm86 for compute capability 8.6
# and 8.9 (sm_86 cubins run on Ada), sm120a for 12.0 -- and runs one of:
#
#   run <model> [profile]     scripts/run.sh: the measured serving profiles (default: run qwen38-27b)
#   download <model>          scripts/download-model.sh into /models
#   ninfer | serve | perplexity | calibrate [args]   the binary itself; serve listens on
#                             NINFER_HOST:NINFER_PORT unless --host/--port are given
#   anything else             executed as given
#
# NINFER_IMAGE_ARCH=sm86|sm120a skips detection.
set -euo pipefail
home=/opt/ninfer
: "${NINFER_HOST:=0.0.0.0}" "${NINFER_PORT:=8080}"

select_arch() {
  if [[ -n "${NINFER_IMAGE_ARCH:-}" ]]; then
    printf '%s' "$NINFER_IMAGE_ARCH"
    return
  fi
  local capability
  if ! capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)" ||
     [[ -z "$capability" ]]; then
    printf 'No GPU visible in the container. Start it with --gpus all (docker run) or a GPU\n' >&2
    printf 'reservation (compose), with the NVIDIA Container Toolkit installed on the host.\n' >&2
    exit 1
  fi
  case "${capability// /}" in
    8.6|8.7|8.9) printf 'sm86' ;;
    12.0) printf 'sm120a' ;;
    *)
      printf 'Compute capability %s is not supported: NInfer runs on 8.6 (RTX 30), 8.9 (RTX 40)\n' "$capability" >&2
      printf 'and 12.0 (RTX 50). NINFER_IMAGE_ARCH=sm86|sm120a overrides the detection.\n' >&2
      exit 1 ;;
  esac
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
  *) exec "$command" "$@" ;;
esac
