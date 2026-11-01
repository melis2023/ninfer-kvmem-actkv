#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
fixture="$root/models/cpu-q2-layer-12.bin"
oracle="$NINFER_JOB_DIR/oracle.bin"
g++ -std=c++20 -O3 -mavx2 -mf16c -fopenmp bench/ops/q2_cpu_probe.cpp \
    -o "$NINFER_JOB_DIR/cpu-oracle"
OMP_PLACES=cores OMP_PROC_BIND=true OMP_WAIT_POLICY=PASSIVE \
    taskset -c "$(cat /sys/devices/system/node/node0/cpulist)" \
    "$NINFER_JOB_DIR/cpu-oracle" "$fixture" --oracle-output "$oracle"
exec "$root/build/bench/ninfer_benches" ninfer_q2_gpu_probe "$fixture" "$oracle"
