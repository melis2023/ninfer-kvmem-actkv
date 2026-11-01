#!/usr/bin/env bash
# Package the configured Linux build with CUDA, architecture and commit labels.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3.11 "$repo_root/scripts/package-linux-release.py"
