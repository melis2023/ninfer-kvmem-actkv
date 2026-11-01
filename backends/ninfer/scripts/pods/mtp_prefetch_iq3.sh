#!/usr/bin/env bash
set -Eeuo pipefail
export HF_TOKEN_PATH=/run/ninfer-hf/token
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_XET_CACHE=/workspace/ninfer-work/mtp-release/xet
/workspace/ninfer-work/py311/bin/hf download \
    WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3 \
    Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
    --revision 5321a4b6f7daf7f14d6408b5905adf8179460830 \
    --local-dir /workspace/ninfer-work/mtp-release/iq3/source
