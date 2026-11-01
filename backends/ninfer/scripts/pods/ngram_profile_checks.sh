#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
cd "$root/src"
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -pthread \
    -I include -I src src/models/qwen4_exp/ngram_table.cpp src/core/file_read_queue.cpp \
    src/models/qwen4_exp/read_pool.cpp src/models/qwen4_exp/ngram_hash.cpp \
    src/models/qwen4_exp/ngram_profile.cpp tests/models/qwen4_exp/test_ngram_table.cpp \
    -o "$NINFER_JOB_DIR/ngram-profile-test"
cd "$NINFER_JOB_DIR"
"$NINFER_JOB_DIR/ngram-profile-test"
