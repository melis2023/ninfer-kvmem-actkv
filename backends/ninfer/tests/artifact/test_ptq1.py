from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import encode_row_split, decode_row_split_codes, dequantize_row_split, split_row_planes
from tools.artifact.layouts import row_split_geometry
from tools.convert.sources.gguf import encode_ptq1_0, decode_ptq1_0


def test_ptq1_matches_independent_gguf_codec():
    rng = np.random.default_rng(17411)
    codes = rng.integers(-1, 2, size=(11, 5, 128), dtype=np.int8)
    scales = (rng.standard_normal((11, 5)) * 0.1).astype(np.float16)
    # Exercise zero and the smallest binary16 subnormal, along with negative multipliers.
    scales[0, :3] = [0, np.nextafter(np.float16(0), np.float16(1)), -0.125]
    blocks = encode_ptq1_0(codes, scales)
    payload = encode_row_split(torch.from_numpy(codes), torch.from_numpy(scales),
                               "ptq1_g128_fp16", (11, 640))
    planes = split_row_planes(payload, row_split_geometry("ptq1_g128_fp16", (11, 640)))
    assert bytes(planes.base) == blocks[:, :, :24].tobytes()
    assert bytes(planes.high) == blocks[:, :, 24:26].tobytes()
    assert bytes(planes.scale) == blocks[:, :, 26:].tobytes()
    actual_scales, actual_codes = decode_row_split_codes(payload, "ptq1_g128_fp16", (11, 640))
    assert torch.equal(actual_codes, torch.from_numpy(codes))
    assert torch.equal(actual_scales.view(torch.int16), torch.from_numpy(scales).view(torch.int16))
    expected = torch.from_numpy(decode_ptq1_0(blocks).values).float() * torch.from_numpy(scales).float().unsqueeze(-1)
    assert torch.equal(dequantize_row_split(payload, "ptq1_g128_fp16", (11, 640)), expected.reshape(11, 640).bfloat16())


def test_ptq1_all_canonical_byte_values():
    # Five base-3 digits have 243 possibilities; the two tail bytes each have four digits.
    values = np.zeros((243, 1, 128), dtype=np.int8)
    for row in range(243):
        for stage in range(5):
            values[row, 0, stage * 16: (stage + 1) * 16] = (row // 3 ** (4 - stage)) % 3 - 1
            values[row, 0, 80 + stage * 8:88 + stage * 8] = (row // 3 ** (4 - stage)) % 3 - 1
        for stage in range(4):
            values[row, 0, 120 + stage * 2:122 + stage * 2] = (row % 81 // 3 ** (3 - stage)) % 3 - 1
    scale = torch.ones((243, 1), dtype=torch.float16)
    encoded = encode_row_split(torch.from_numpy(values), scale, "ptq1_g128_fp16", (243, 128))
    _, decoded = decode_row_split_codes(encoded, "ptq1_g128_fp16", (243, 128))
    assert torch.equal(decoded, torch.from_numpy(values))


def test_ptq1_geometry_and_invalid_domain():
    g = row_split_geometry("ptq1_g128_fp16", (7, 5120))
    assert g.base_bytes_per_group == 24 and g.high_bytes_per_group == 2
    assert g.payload_bytes == g.scale_offset + 7 * 40 * 2
    with pytest.raises(ValueError, match="128"):
        row_split_geometry("ptq1_g128_fp16", (3, 129))
    with pytest.raises(ValueError, match="PTQ1 codes"):
        encode_row_split(torch.full((1,1,128), 2, dtype=torch.int8),
                         torch.ones((1,1), dtype=torch.float16), "ptq1_g128_fp16", (1,128))
