#!/usr/bin/env bash
set -Eeuo pipefail
# A bounded observer: samples at 0/30/60/90/120 seconds. No command arguments or environment.
for sample in 0 1 2 3 4; do
    date -u +%FT%TZ
    uptime
    free -m
    for metric in memory.current memory.max memory.events cpu.max cpu.stat memory.pressure; do
        if [ -r "/sys/fs/cgroup/$metric" ]; then
            printf 'cgroup %s\n' "$metric"
            cat "/sys/fs/cgroup/$metric"
        fi
    done
    df -h /workspace
    cat /proc/net/dev
    for input in /workspace/ninfer-work/jobs/*/inputs/*.tar.gz; do
        [ ! -f "$input" ] || stat -c 'job_input %n %s bytes' "$input"
    done
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
            --format=csv,noheader || true
    fi
    ps -eo pid,ppid,comm,etimes,pcpu,rss --sort=-rss | head -n 16 || true
    for process in $(pgrep -x python || true) $(pgrep -x ninfer_tests || true) \
                   $(pgrep -x ninfer || true) $(pgrep -x ninfer-serve || true); do
        printf 'process_io pid=%s\n' "$process"
        cat "/proc/$process/io" 2>/dev/null || true
    done
    for build in /workspace/ninfer-work/build; do
        [ ! -f "$build/.ninja_log" ] || wc -l "$build/.ninja_log"
    done
    [ "$sample" -eq 4 ] || sleep 30
done
