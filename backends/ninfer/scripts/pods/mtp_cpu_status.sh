#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
date -u +%FT%TZ
for directory in "$root/models" "$root/mtp-release" "$root/mtp-candidates"; do
    [ -d "$directory" ] || continue
    find "$directory" -maxdepth 8 -type f \
        \( -name '*.ninfer' -o -name '*.conversion.json' -o -name '*.incomplete' -o -name '*.partial' \) \
        -printf '%p %s bytes %b allocated-blocks\n'
done
df -h "$root"
ps -eo pid,ppid,comm,etimes,pcpu,rss --sort=-rss | head -n 12 || true
