#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export NINFER_QWEN4_EXP_ARTIFACT="$root/models/flash-next-native-q2-mtp.ninfer"
export NINFER_QWEN4_EXP_NGRAM_TABLE="$root/models/flash-next-iq4-table.ninfer"
export NINFER_QWEN4_EXP_DRAFTS=0
export NINFER_QWEN4_EXP_PREFILL_CHUNK=512
export NINFER_QWEN4_EXP_LOGITS_REPORT="$NINFER_JOB_DIR/logits.json"
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 2 ]; then
    export NINFER_QWEN4_EXP_DEVICES=0,1
    export NINFER_QWEN4_EXP_EXPERTS=device
    "$root/build/tests/ninfer_tests" ninfer_qwen4_exp_generate_real
else
    export NINFER_QWEN4_EXP_DEVICES=0
    export NINFER_QWEN4_EXP_EXPERTS=host
    export NINFER_QWEN4_EXP_EXPERT_CACHE_MIB=4096
    for share in 0 0.5 1; do
        export NINFER_QWEN4_EXP_DMA_SHARE="$share"
        export NINFER_QWEN4_EXP_LOGITS_REPORT="$NINFER_JOB_DIR/logits-dma-$share.json"
        "$root/build/tests/ninfer_tests" ninfer_qwen4_exp_generate_real
    done
fi
