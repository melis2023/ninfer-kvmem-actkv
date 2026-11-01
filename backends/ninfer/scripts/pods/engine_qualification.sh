#!/usr/bin/env bash
set -u
failed=0
for routine in engine_checks cache_checks sampling_peer_checks; do
    bash "scripts/pods/$routine.sh"
    rc=$?
    printf 'qualification routine=%s exit=%s\n' "$routine" "$rc"
    if [ "$rc" != 0 ]; then failed=1; fi
done
exit "$failed"
