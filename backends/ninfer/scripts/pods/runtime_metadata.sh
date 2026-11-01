#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
lscpu -J > "$NINFER_JOB_DIR/cpu.json"
nvidia-smi -q -x > "$NINFER_JOB_DIR/gpu.xml"
nvidia-smi topo -m > "$NINFER_JOB_DIR/topology.txt"
profile="${NINFER_DEVICE_PROFILES:-${XDG_CACHE_HOME:-$HOME/.cache}/ninfer/device-profiles.json}"
if [ -f "$profile" ]; then cp "$profile" "$NINFER_JOB_DIR/device-profiles.json"; fi
cp "$root/src/src/runtime/engine/device_profiles.json" "$NINFER_JOB_DIR/builtin-profiles.json"
if [ -f "$root/models/config/generation_config.json" ]; then
    cp "$root/models/config/generation_config.json" "$NINFER_JOB_DIR/generation_config.json"
fi
echo RUNTIME_METADATA_READY
