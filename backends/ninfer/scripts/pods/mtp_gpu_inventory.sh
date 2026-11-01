#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
date -u +%FT%TZ
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
free -m
df -h "$root"
[ ! -d "$root/models" ] || find "$root/models" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
find "$root" -maxdepth 4 -type f \( -name ninfer-serve -o -name ninfer-ngram-profile \
    -o -name CMakeCache.txt \) -printf '%p %s bytes\n'
cat "$root/source.json"
nvcc --version
"$root/py311/bin/python" --version
"$root/py311/bin/python" -c 'import huggingface_hub; print("huggingface_hub", huggingface_hub.__version__)'
