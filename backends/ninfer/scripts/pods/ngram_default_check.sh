#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
ctest --test-dir "$root/build" --output-on-failure -R '^ninfer_cli_options_test$'
"$root/build/apps/ninfer" "$root/models/flash-next-native-q2-mtp.ninfer" --devices 0,1 \
    --ngram-table "$root/models/flash-next-q2_0-mtp.ninfer" --greedy --no-thinking \
    --max-context 4096 --kv-capacity 4096 --kv-dtype int8 --max-new 1 --prompt 'Compute 2+2.' \
    > "$NINFER_JOB_DIR/default.txt" 2> "$NINFER_JOB_DIR/default.err"
grep -q 'row cache capacity 0 MiB' "$NINFER_JOB_DIR/default.err"
grep -Eq 'ngram rows from RAM[[:space:]]+0\.0%' "$NINFER_JOB_DIR/default.err"
echo NGRAM_DEFAULT_PASS
