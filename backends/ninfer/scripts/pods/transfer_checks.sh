#!/usr/bin/env bash
set -Eeuo pipefail
exec taskset -c "$(cat /sys/devices/system/node/node0/cpulist)" \
    /workspace/ninfer-work/py311/bin/python "$NINFER_JOB_DIR/inputs/transfer_probe.py"
