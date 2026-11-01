"""Exact E4M3FN words with BF16 scale planes or interleaved binary16 scales."""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import row_interleaved_geometry, row_scale_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)
from .direct import decode_direct, encode_direct


def _exact_bf16_vector(tensor: torch.Tensor, length: int, label: str) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (length,):
        raise TypeError(f"{label} must be BF16 with shape ({length},)")
    return tensor.detach().contiguous().cpu()


def validate_fp8_row_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if bool(((codes & 0x7F) == 0x7F).any()):
        raise ValueError("row-scaled FP8 codes must be finite E4M3FN words")
    if scales.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("row-scaled FP8 scales must be BF16 or FP16")
    scale_words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = 0x7F80 if scales.dtype == torch.bfloat16 else 0x7C00
    invalid_scales = ((scale_words & 0x8000) != 0) | ((scale_words & exponent) == exponent)
    if bool(invalid_scales.any()):
        name = "BF16" if scales.dtype == torch.bfloat16 else "FP16"
        raise ValueError(f"row-scaled FP8 scales must be nonnegative finite {name} words")
    zero_scale = scale_words == 0
    nonzero_code = (codes & 0x7F) != 0
    if bool((zero_scale.unsqueeze(1) & nonzero_code).any()):
        raise ValueError("a zero row scale requires only signed-zero FP8 codes")


def encode_fp8_row_scaled(
    code_words: torch.Tensor,
    row_scales: torch.Tensor,
    shape: Sequence[int],
) -> bytes:
    """Encode exact E4M3FN code words and BF16 row multipliers."""

    geometry = row_scale_geometry("fp8_e4m3fn_row_bf16", shape)
    codes = _exact_uint8_matrix(
        code_words,
        (geometry.n, geometry.k),
        "row-scaled FP8 codes",
    )
    scales = _exact_bf16_vector(
        row_scales,
        geometry.n,
        "row-scaled FP8 scales",
    )
    validate_fp8_row_words(codes, scales)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    scale_begin = geometry.scale_plane_offset
    payload[scale_begin : scale_begin + geometry.scale_plane_bytes] = encode_direct(
        scales, "bf16"
    )
    return bytes(payload)


def decode_fp8_row_scaled_words(
    payload: Payload,
    shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode exact E4M3FN code words and BF16 row multipliers."""

    geometry = row_scale_geometry("fp8_e4m3fn_row_bf16", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"row-scaled FP8 payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].clone().reshape(geometry.n, geometry.k)
    scale_begin = geometry.scale_plane_offset
    scale_bytes = raw[scale_begin : scale_begin + geometry.scale_plane_bytes]
    scales = decode_direct(scale_bytes, "bf16", (geometry.n,))
    validate_fp8_row_words(codes, scales)
    return codes, scales


def encode_fp8_row_interleaved(
    code_words: torch.Tensor, row_scales: torch.Tensor, shape: Sequence[int]
) -> bytes:
    """Store each row's K E4M3FN bytes immediately before its little-endian FP16 scale."""
    g = row_interleaved_geometry("fp8_e4m3fn_row_fp16", shape)
    codes = _exact_uint8_matrix(code_words, (g.n, g.k), "row-scaled FP8 codes")
    if row_scales.dtype != torch.float16 or tuple(row_scales.shape) != (g.n,):
        raise TypeError(f"row-scaled FP8 scales must be FP16 with shape ({g.n},)")
    scales = row_scales.detach().contiguous().cpu()
    validate_fp8_row_words(codes, scales)
    raw = torch.empty((g.n, g.row_bytes), dtype=torch.uint8)
    raw[:, :g.k] = codes
    # Explicit byte order, independent of the host tensor's word representation.
    words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
    raw[:, g.k] = (words & 255).to(torch.uint8)
    raw[:, g.k + 1] = (words >> 8).to(torch.uint8)
    return raw.numpy().tobytes()


def decode_fp8_row_interleaved_words(
    payload: Payload, shape: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    g = row_interleaved_geometry("fp8_e4m3fn_row_fp16", shape)
    if _payload_length(payload) != g.payload_bytes:
        raise ValueError(f"interleaved FP8 payload must contain {g.payload_bytes} bytes")
    raw = _payload_tensor(payload, torch.device("cpu")).reshape(g.n, g.row_bytes)
    codes = raw[:, :g.k].clone()
    words = raw[:, g.k].to(torch.int32) | (raw[:, g.k + 1].to(torch.int32) << 8)
    scales = words.to(torch.int16).view(torch.float16)
    validate_fp8_row_words(codes, scales)
    return codes, scales


def dequantize_fp8_row_scaled(
    payload: Payload,
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a row-scaled FP8 matrix from its exact stored words."""

    codes, scales = decode_fp8_row_scaled_words(payload, shape)
    return (codes.view(torch.float8_e4m3fn).float() * scales.float().unsqueeze(1)).to(
        dtype
    )
