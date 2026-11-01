#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
py="$root/py311/bin/python"
model=$("$py" -c 'import json; print(json.load(open("/workspace/ninfer-work/mtp-candidates/q2/download-verified.json"))["model_path"])')
"$root/build/apps/ninfer-ngram-profile" "$model" \
    --out "$root/ngram-profile/broad-v1/native-reference.hot" \
    "$root/ngram-corpus/broad-v1/train.jsonl"
cmp "$root/ngram-profile/broad-v1/broad-v1.hot" "$root/ngram-profile/broad-v1/native-reference.hot"
"$root/build/apps/ninfer-ngram-profile" "$model" \
    --evaluate "$root/ngram-profile/broad-v1/broad-v1.hot" --row-bytes 90 \
    "$root/ngram-corpus/broad-v1/heldout.jsonl"
sha256sum "$root/ngram-profile/broad-v1/broad-v1.hot" > "$NINFER_JOB_DIR/profile.sha256"
echo NATIVE_PROFILE_EXACT_PASS
