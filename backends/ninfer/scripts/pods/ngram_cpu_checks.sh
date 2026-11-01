#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread -I "$root/src/src" \
    "$root/src/src/core/file_read_queue.cpp" "$root/src/tests/test_file_read_queue.cpp" \
    -o "$NINFER_JOB_DIR/file-read-queue-test"
cd "$NINFER_JOB_DIR"
set +e
./file-read-queue-test
queue_status=$?
set -e
if [ "$queue_status" -ne 0 ] && [ "$queue_status" -ne 77 ]; then exit "$queue_status"; fi
# The reader is CPU-only; compile its production sources directly while the CUDA bundle builds.
reader="$root/src/src/models/qwen4_exp/ngram_table.cpp"
if [ -f "$NINFER_JOB_DIR/inputs/ngram_table.cpp" ]; then
    reader="$NINFER_JOB_DIR/inputs/ngram_table.cpp"
fi
g++ -std=c++20 -O2 -pthread -I "$root/src/include" -I "$root/src/src" \
    "$reader" "$root/src/src/core/file_read_queue.cpp" \
    "$root/src/src/models/qwen4_exp/read_pool.cpp" \
    "$root/src/src/models/qwen4_exp/ngram_hash.cpp" \
    "$root/src/src/models/qwen4_exp/ngram_profile.cpp" \
    "$root/src/tests/models/qwen4_exp/test_ngram_table.cpp" -o "$NINFER_JOB_DIR/ngram-table-test"
cd "$NINFER_JOB_DIR"
./ngram-table-test
