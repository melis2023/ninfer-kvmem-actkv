#!/usr/bin/env bash
# The artifacts the decision checks serve: the published Ternary Bonsai 2 27B (a Qwen3.5-core
# model), Gemma 4 26B-A4B (QAT Q4_0 with MTP) and Qwen3.8-Flash-Next (Q2_0) with its n-gram table.
# Arguments select the models (bonsai, gemma4, flash-next); none selects all three.
set -Eeuo pipefail
root=/workspace/ninfer-work
hf="$root/py311/bin/hf"
fetch() { # repo file
    mkdir -p "$root/models"
    [ -f "$root/models/$2" ] || "$hf" download "$1" "$2" --local-dir "$root/models"
    ls -l "$root/models/$2"
}
wanted=" ${*:-bonsai gemma4 flash-next} "
case "$wanted" in *" bonsai "*)
    fetch WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 Ternary-Bonsai-2-27B-ninfer-v3.ninfer ;;
esac
case "$wanted" in *" gemma4 "*)
    fetch WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3 gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer ;;
esac
case "$wanted" in *" flash-next "*)
    fetch WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3 Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer
    fetch WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3 Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer ;;
esac
