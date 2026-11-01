#!/usr/bin/env bash
set -Eeuo pipefail
lscpu
for node in /sys/devices/system/node/node*; do
    printf '%s cpus=' "${node##*/}"
    cat "$node/cpulist"
done
nvidia-smi topo -m
df -h /workspace
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max
