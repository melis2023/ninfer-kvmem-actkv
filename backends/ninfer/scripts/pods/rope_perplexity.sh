#!/usr/bin/env bash
# Perplexity past the native window with YaRN and without, for Qwen3.8-27B, Qwen3.8-Flash-Next and
# Gemma 4, once the models are down; the reports land under the job directory.
set -Eeuo pipefail
root=/workspace/ninfer-work
while [ ! -f "$root/jobs/models/exit" ]; do sleep 5; done
[ "$(cat "$root/jobs/models/exit")" = 0 ] || exit 125
"$root/py311/bin/python" scripts/pods/rope_perplexity.py --build "$root/build" \
    --models "$root/models" --corpus eval/corpora/perplexity-1m --out "$NINFER_JOB_DIR/perplexity" "$@"
