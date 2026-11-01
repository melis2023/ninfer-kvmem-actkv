#!/usr/bin/env bash
set -Eeuo pipefail
bash scripts/pods/engine_checks.sh
bash scripts/pods/ngram_checks.sh
echo FLASH_CHECKS_PASS
