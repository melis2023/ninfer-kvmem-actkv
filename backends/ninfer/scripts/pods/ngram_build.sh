#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
baseline="$root/baselines/ngram-pre-cache-20261008"
if [ ! -d "$baseline" ]; then
    mkdir -p "$baseline"
    cp "$root/build/apps/ninfer" "$baseline/ninfer"
fi
# The build runner has already replaced jobs/build with this new snapshot. The previous
# binary was qualified by native-a16-checks; preserve that job's source identity instead.
cp "$root/jobs/native-a16-checks/source.json" "$baseline/source.json"
bash scripts/pods/build.sh --target ninfer ninfer_tests
