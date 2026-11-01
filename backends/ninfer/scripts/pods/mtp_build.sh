#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
bash scripts/pods/build.sh --target ninfer ninfer-serve ninfer-ngram-profile
mkdir -p "$root/confidence-runtime/apps"
ln -sfn "$root/build/apps/ninfer-serve" "$root/confidence-runtime/apps/ninfer-serve"
cp "$root/source.json" "$root/confidence-runtime/source.json"
# Reuse the completed Op and draft-control qualifications. This stage builds the public
# executables for the new artifact checks instead of recompiling the full test dispatcher.
echo MTP_RUNTIME_BUILD_PASS
