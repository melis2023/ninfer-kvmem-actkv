#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
fixture="$root/models/cpu-q2-layer-12.bin"
# The retained oracle independently decoded these same stored weights and eight BF16 columns.
oracle="$root/jobs/native-probe/oracle.bin"
test -f "$oracle"
sha256sum "$fixture" "$oracle" > "$NINFER_JOB_DIR/inputs.sha256"
failed=0
for route in gguf native-a16 native-a8; do
    if "$root/build/bench/ninfer_benches" ninfer_q2_gpu_probe "$fixture" "$oracle" "$route" \
        >"$NINFER_JOB_DIR/$route.log" 2>&1; then
        cat "$NINFER_JOB_DIR/$route.log"
    else
        cat "$NINFER_JOB_DIR/$route.log"
        failed=1
    fi
done
exit "$failed"
