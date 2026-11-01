#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
export PYTHONPATH="$root/src"
prefix=model.language_model.layers.0.mlp.experts.0
"$root/py311/bin/python" -m tools.reference.fetch_slice \
    --repo Qwen/Qwen3.8-Flash-Next-FP8 --revision 236dfdf285828023ca3bcd3f37366c58a3469b13 \
    --out "$NINFER_JOB_DIR/subset" \
    --name "$prefix.gate_proj.weight" --name "$prefix.gate_proj.weight_scale_inv" \
    --name "$prefix.up_proj.weight" --name "$prefix.up_proj.weight_scale_inv" \
    --name "$prefix.down_proj.weight" --name "$prefix.down_proj.weight_scale_inv"
