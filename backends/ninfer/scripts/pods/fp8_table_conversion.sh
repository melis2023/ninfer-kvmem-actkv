#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
source_dir="$root/sources/flash-next-bf16-ngram"
output="$root/models/flash-next-fp8-table.ninfer"
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
if [ ! -f "$source_dir/fetch-manifest.json" ]; then
    "$root/py311/bin/python" -m tools.reference.fetch_slice \
        --repo Qwen/Qwen3.8-Flash-Next \
        --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
        --prefix model.language_model.layers.1.ple.ple_embedding.ngram_embedding. \
        --out "$source_dir" --workers 8
fi
if [ ! -f "$output" ]; then
    "$root/py311/bin/python" -m tools.convert --model "$source_dir" \
        --recipe qwen3_8_flash_next --components ngram --source "ngram=$source_dir" \
        --device cpu --rows-per-chunk 65536 --name flash-next-fp8-table --out "$output"
fi
"$root/py311/bin/python" -m tools.artifact.inspect "$output" --json
