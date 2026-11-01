#!/usr/bin/env bash
set -Eeuo pipefail
export NINFER_FLASH_NEXT_DRAFTS=4
bash scripts/pods/long_context_checks.sh
