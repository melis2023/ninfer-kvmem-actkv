#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
fixture="$root/models/cpu-q2-layer-12.bin"
oracle="$root/jobs/native-probe/oracle.bin"
baseline="$root/baselines/native-a16-vector-20261008/ninfer_benches"
if [ ! -f "$baseline" ] && [ -f "$baseline.gz" ]; then
    gzip -dk "$baseline.gz"
    chmod +x "$baseline"
fi
test -f "$baseline"
test -f "$oracle"
ctest --test-dir "$root/build" --output-on-failure -V -R '^ninfer_moe_experts_native_test$'
"$baseline" ninfer_q2_gpu_probe "$fixture" "$oracle" native-a16 \
    > "$NINFER_JOB_DIR/baseline-a16.log" 2>&1
tail -n 1 "$NINFER_JOB_DIR/baseline-a16.log"
for route in gguf native-a16 native-a8; do
    "$root/build/bench/ninfer_benches" ninfer_q2_gpu_probe "$fixture" "$oracle" "$route" \
        > "$NINFER_JOB_DIR/$route.log" 2>&1
    tail -n 1 "$NINFER_JOB_DIR/$route.log"
done
bash scripts/pods/native_engine_checks.sh
