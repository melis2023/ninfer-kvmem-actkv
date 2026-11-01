"""Decode HF E4M3FN matrices with explicit 128x128 block multipliers."""
from __future__ import annotations

import torch

from .logical import LogicalSource
from .safetensors import SafetensorsSource


def block_fp8_matrix_source(
    store: SafetensorsSource, name: str, shape: tuple[int, int]
) -> LogicalSource:
    config = store.config.get("quantization_config", {})
    if config.get("quant_method") != "fp8" or config.get("weight_block_size") != [128, 128]:
        raise ValueError(f"{name}: block FP8 requires the HF fp8 128x128 weight contract")
    n, k = shape
    weight = store.describe(name)
    scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
    scale = store.describe(scale_name)
    scale_shape = ((n + 127) // 128, (k + 127) // 128)
    if weight.dtype != "F8_E4M3" or weight.shape != shape:
        raise ValueError(f"{name}: expected F8_E4M3{shape}, got {weight.dtype}{weight.shape}")
    if scale.dtype not in ("BF16", "F32") or scale.shape != scale_shape:
        raise ValueError(f"{scale_name}: expected BF16/F32{scale_shape}, got {scale.dtype}{scale.shape}")

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        codes = store.read_flat(name, begin, end)
        words = codes.view(torch.uint8)
        if bool(((words & 0x7F) == 0x7F).any()):
            raise ValueError(f"{name}: block FP8 weight codes must be finite")
        first = (begin // k) // 128
        last = ((end - 1) // k) // 128 + 1
        scales = store.read_flat(scale_name, first * scale_shape[1], last * scale_shape[1]).float()
        if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
            raise ValueError(f"{scale_name}: block scales must be finite and nonnegative")
        index = torch.arange(begin, end, dtype=torch.int64)
        scale_index = (index // k // 128 - first) * scale_shape[1] + index % k // 128
        # Despite the HF name weight_scale_inv, these words are dequantization multipliers.
        # Preserve their FP32 precision; the selected conversion method owns subsequent casts.
        values = codes.float() * scales[scale_index]
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name}: decoded FP8 values overflow FP32")
        return values

    return LogicalSource(shape, f"{store.path}:{name} (HF block FP8)", read)
