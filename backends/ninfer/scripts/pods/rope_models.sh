#!/usr/bin/env bash
# The artifacts the rope-scaling perplexity runs score, one directory each: Qwen3.8-27B (IQ3_S),
# Gemma 4 26B-A4B (QAT Q4_0 with MTP), and Qwen3.8-Flash-Next (IQ3_S) with its n-gram table.
# Arguments select the models (qwen38-27b, gemma4, flash-next); none selects all three.
set -Eeuo pipefail
root=/workspace/ninfer-work
hf="$root/py311/bin/hf"
fetch() { # repo file directory
    mkdir -p "$root/models/$3"
    [ -f "$root/models/$3/$2" ] || "$hf" download "$1" "$2" --local-dir "$root/models/$3"
    ls -l "$root/models/$3/$2"
}
wanted=" ${*:-qwen38-27b gemma4 flash-next} "
case "$wanted" in *" qwen38-27b "*)
    fetch WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3 Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer qwen38-27b ;;
esac
case "$wanted" in *" gemma4 "*)
    fetch WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3 gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer gemma4 ;;
esac
case "$wanted" in *" flash-next "*)
    fetch WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3 Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer flash-next
    fetch WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3 Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-ninfer-v3.ninfer flash-next ;;
esac
