from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    encoded_size,
    row_scale_geometry,
)
from tools.artifact.codecs.fp8_row import (
    decode_fp8_row_interleaved_words,
    decode_fp8_row_scaled_words,
    dequantize_fp8_row_scaled,
    encode_fp8_row_scaled,
    encode_fp8_row_interleaved,
)


def _bf16_words(*words: int) -> torch.Tensor:
    signed = [word if word < 0x8000 else word - 0x10000 for word in words]
    return torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16)


def test_row_scale_layout_known_words_padding_and_reconstruction():
    shape = (2, 4)
    geometry = row_scale_geometry("fp8_e4m3fn_row_bf16", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.payload_bytes,
    ) == (8, 256, 4, 260)
    assert encoded_size("row_scale_v1", "fp8_e4m3fn_row_bf16", shape) == 260

    codes = torch.tensor(
        [
            [0x00, 0x80, 0x38, 0xB8],
            [0x40, 0xC0, 0x7E, 0xFE],
        ],
        dtype=torch.uint8,
    )
    scales = _bf16_words(0x3F00, 0x4000)  # 0.5, 2.0
    payload = encode_fp8_row_scaled(codes, scales, shape)

    assert payload[:8] == bytes(codes.reshape(-1).tolist())
    assert payload[8:256] == bytes(248)
    assert payload[256:] == struct.pack("<HH", 0x3F00, 0x4000)

    decoded_codes, decoded_scales = decode_fp8_row_scaled_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales.view(torch.int16), scales.view(torch.int16))
    assert torch.equal(
        dequantize_fp8_row_scaled(payload, shape),
        torch.tensor(
            [
                [0.0, -0.0, 0.5, -0.5],
                [4.0, -4.0, 896.0, -896.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_row_scaled_fp8_rejects_invalid_words_and_signatures():
    zero_codes = torch.zeros((1, 2), dtype=torch.uint8)
    positive_scale = _bf16_words(0x3F80)

    invalid_codes = zero_codes.clone()
    invalid_codes[0, 0] = 0x7F
    with pytest.raises(ValueError, match="finite E4M3FN"):
        encode_fp8_row_scaled(invalid_codes, positive_scale, (1, 2))

    with pytest.raises(ValueError, match="nonnegative finite BF16"):
        encode_fp8_row_scaled(zero_codes, _bf16_words(0x8000), (1, 2))

    nonzero_codes = zero_codes.clone()
    nonzero_codes[0, 0] = 0x38
    with pytest.raises(ValueError, match="zero row scale"):
        encode_fp8_row_scaled(nonzero_codes, _bf16_words(0x0000), (1, 2))


def test_interleaved_fp8_exact_bytes_and_invalid_words():
    codes = torch.tensor([[0, 128, 56, 184], [64, 192, 126, 254]], dtype=torch.uint8)
    scales = torch.tensor([.5, 2], dtype=torch.float16)
    raw = encode_fp8_row_interleaved(codes, scales, (2, 4))
    assert raw == bytes([0, 128, 56, 184, 0, 56, 64, 192, 126, 254, 0, 64])
    decoded, restored = decode_fp8_row_interleaved_words(raw, (2, 4))
    assert torch.equal(decoded, codes) and torch.equal(restored, scales)
    assert encoded_size("row_interleaved_v1", "fp8_e4m3fn_row_fp16", (2, 160)) == 324
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row_scale_v1", "fp8_e4m3fn_row_fp16", (2, 160))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row_interleaved_v1", "fp8_e4m3fn_row_bf16", (2, 160))
    for scale in (-0., -1., float("inf"), float("nan")):
        with pytest.raises(ValueError, match="nonnegative finite FP16"):
            encode_fp8_row_interleaved(codes[:1], torch.tensor([scale], dtype=torch.float16), (1, 4))
    with pytest.raises(ValueError, match="zero row scale"):
        decode_fp8_row_interleaved_words(bytes([56, 0, 0]), (1, 1))
    for word in (127, 255):
        with pytest.raises(ValueError, match="finite E4M3FN"):
            decode_fp8_row_interleaved_words(bytes([word, 0, 60]), (1, 1))
    with pytest.raises(ValueError, match="bytes"):
        decode_fp8_row_interleaved_words(raw[:-1], (2, 4))
