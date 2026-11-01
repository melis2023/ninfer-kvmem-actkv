#!/usr/bin/env bash
set -Eeuo pipefail
if ! command -v gdb >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq gdb
fi
process=$(pgrep -x ninfer_tests | head -n 1)
gdb -batch -p "$process" -ex 'set pagination off' -ex 'thread apply all bt 12' -ex detach
