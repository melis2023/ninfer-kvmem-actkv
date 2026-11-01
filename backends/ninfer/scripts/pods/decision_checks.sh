#!/usr/bin/env bash
# The typed decision endpoints on a GPU: the readout Op and serve tests, then the served model's
# answers, their agreement with the chat endpoint's log probabilities, latency by state length
# and question count, and the endpoint beside a concurrent chat request.
set -Eeuo pipefail
root=/workspace/ninfer-work
build="$root/build"
ctest --test-dir "$build" --output-on-failure -j2 \
    -R 'ninfer_(readout_logprobs|target_logprobs|logprob_topk|systemone|rerank)_test'
while [ ! -f "$root/jobs/models/exit" ]; do sleep 2; done
[ "$(cat "$root/jobs/models/exit")" = 0 ] || exit 125
"$root/py311/bin/python" scripts/pods/decision_bench.py "$@"
