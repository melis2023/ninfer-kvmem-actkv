from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.codecs.row_split import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
)


def test_q2_g64_preserves_offset_codes_and_signed_binary16_scales():
    # Two rows, two groups per row. Every adjacent quartet is -1, 0, 1, 2:
    # its offset codes 0, 1, 2, 3 must form literal byte 0xe4.
    codes = torch.tensor([-1, 0, 1, 2] * 64, dtype=torch.int8).reshape(2, 2, 64)
    scale_words = (0x3800, 0xBC00, 0x0001, 0x8000)
    scales = torch.tensor(scale_words, dtype=torch.uint16).view(torch.float16).reshape(2, 2)
    expected = bytearray(264)
    expected[:64] = b"\xe4" * 64
    expected[256:] = struct.pack("<4H", *scale_words)
    payload = encode_row_split(codes, scales, "q2_g64_fp16", (2, 128))
    assert payload == expected
    actual_scales, actual_codes = decode_row_split_codes(payload, "q2_g64_fp16", (2, 128))
    assert torch.equal(actual_codes, codes)
    assert torch.equal(actual_scales.view(torch.uint16), scales.view(torch.uint16))
    oracle = torch.tensor(
        [(-1, 0, 1, 2)[i % 4] * struct.unpack("<e", struct.pack("<H", scale_words[i // 64]))[0]
         for i in range(256)], dtype=torch.float64,
    ).reshape(2, 128)
    values = dequantize_row_split(payload, "q2_g64_fp16", (2, 128), dtype=torch.float64)
    assert torch.equal(values, oracle)
    assert torch.equal(dequantize_row_split(payload, "q2_g64_fp16", (2, 128)),
                       oracle.to(torch.bfloat16))


@pytest.mark.parametrize("k", [65, 640, 2560])
def test_q2_g64_public_shapes_and_logical_zero_padding(k):
    padded = (k + 127) // 128 * 128
    codes = torch.zeros((3, padded // 64, 64), dtype=torch.int8)
    codes.reshape(3, padded)[:, :k] = torch.arange(k).remainder(4).to(torch.int8) - 1
    scales = torch.full((3, padded // 64), 0.25, dtype=torch.float16)
    payload = encode_row_split(codes, scales, "q2_g64_fp16", (3, k))
    _, decoded = decode_row_split_codes(payload, "q2_g64_fp16", (3, k))
    assert torch.equal(decoded, codes)
    expected = (torch.arange(k).remainder(4).double() - 1) * 0.25
    assert torch.equal(dequantize_row_split(payload, "q2_g64_fp16", (3, k), dtype=torch.float64),
                       expected.expand(3, k))


@pytest.mark.parametrize("invalid", [-2, 3])
def test_q2_g64_refuses_codes_outside_the_stored_grid(invalid):
    codes = torch.zeros((1, 2, 64), dtype=torch.int8)
    codes[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="codes"):
        encode_row_split(codes, torch.ones((1, 2), dtype=torch.float16), "q2_g64_fp16", (1, 128))


@pytest.mark.parametrize("invalid", [float("inf"), -float("inf"), float("nan")])
def test_q2_g64_refuses_nonfinite_scale_words(invalid):
    with pytest.raises(ValueError, match="scales"):
        encode_row_split(torch.zeros((1, 2, 64), dtype=torch.int8),
                         torch.full((1, 2), invalid, dtype=torch.float16),
                         "q2_g64_fp16", (1, 128))
