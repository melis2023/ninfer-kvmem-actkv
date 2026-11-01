#!/usr/bin/env bash
set -Eeuo pipefail
lscpu -J > "$NINFER_JOB_DIR/cpu.json"
grep -q avx512_vnni /proc/cpuinfo
if ! command -v g++ >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq g++
fi
bash scripts/pods/cpu_expert_checks.sh | tee "$NINFER_JOB_DIR/oracle.log"
grep -q '^avx512-vnni ' "$NINFER_JOB_DIR/oracle.log"
echo VNNI_ORACLE_PASS
