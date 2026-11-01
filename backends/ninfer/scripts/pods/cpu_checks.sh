#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
while [ ! -f "$root/jobs/checks/exit" ]; do sleep 2; done
# The CPU experiment is independent of the Engine result, but timing needs an idle test host.
exec "$root/py311/bin/python" "$NINFER_JOB_DIR/inputs/cpu_probe.py" \
    "$root/models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" --layer 12
