from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
)
from tools.artifact.formats import get_format
from tools.artifact.layouts import row_split_geometry
from tools.convert.quantization.groupwise import quantize_matrix, search_quantize_matrix


def _half(value: float) -> float:
    """binary64 -> binary32 -> binary16, as Python floats."""
    single = struct.unpack("<f", struct.pack("<f", value))[0]
    try:
        return struct.unpack("<e", struct.pack("<e", single))[0]
    except OverflowError:
        return float("inf") if single > 0 else float("-inf")


def _step(scale: float, offset: int) -> float:
    word = struct.unpack("<H", struct.pack("<e", scale))[0]
    sign, magnitude = word & 0x8000, word & 0x7FFF
    if magnitude == 0 or not 0 < magnitude + offset <= 0x7BFF:
        return 0.0
    return struct.unpack("<e", struct.pack("<H", sign | (magnitude + offset)))[0]


def reference_search(weight, format, importance=None, negative=False):
    """Per-group scalar oracle of the search's candidate list and error criterion.

    Written with Python loops and explicit binary16/binary32 steps, independently of the
    vectorised implementation."""
    spec = get_format(format)
    geometry = row_split_geometry(spec, weight.shape)
    group = spec.group_size
    values = np.zeros((geometry.n, geometry.k_pad), dtype=np.float32)
    values[:, : geometry.k] = weight.float().numpy()
    channel = np.zeros(geometry.k_pad)
    channel[: geometry.k] = 1.0 if importance is None else importance.double().numpy()
    codes = np.zeros((geometry.n, geometry.groups_per_row, group), dtype=np.int8)
    scales = np.zeros((geometry.n, geometry.groups_per_row), dtype=np.float16)

    for row in range(geometry.n):
        for index in range(geometry.groups_per_row):
            w = values[row, index * group : (index + 1) * group]
            imp = channel[index * group : (index + 1) * group]

            def encode(scale):
                reciprocal = np.float32(1.0 / scale) if scale != 0 else np.float32(0.0)
                return np.clip(np.rint(w * reciprocal), spec.qmin, spec.qmax)

            def error(scale, q):
                return float((imp * (w.astype(np.float64) - scale * q) ** 2).sum())

            def fit(q):
                denominator = float((imp * q * q).sum())
                if denominator <= 0:
                    return 0.0
                return _half(float((imp * w.astype(np.float64) * q).sum()) / denominator)

            amax = float(np.abs(w).max())
            if amax == 0:
                continue
            extreme = float(w[int(np.abs(w).argmax())])
            canonical = _half(amax / spec.qmax) or 2.0**-24
            best_scale, best_codes = canonical, encode(canonical)
            best_error = error(best_scale, best_codes)

            def consider(scale):
                nonlocal best_scale, best_codes, best_error
                if scale == 0 or not np.isfinite(scale) or (scale < 0 and not negative):
                    return
                q = encode(scale)
                e = error(scale, q)
                if e < best_error:
                    best_scale, best_codes, best_error = scale, q, e

            ratios = [round(0.70 + 0.02 * step, 2) for step in range(25)]
            for base in (amax / spec.qmax, extreme / spec.qmin):
                for ratio in ratios:
                    consider(_half(base * ratio) or np.copysign(2.0**-24, base))
            for exponent in (-2, -1, 1, 2):
                consider(_half(canonical * 2.0**exponent))
            for offset in (-2, -1, 1, 2):
                consider(_step(canonical, offset))
            least = fit(encode(canonical))
            consider(least)
            for exponent in (-1, 1):
                consider(_half(least * 2.0**exponent))
            for offset in (-2, -1, 1, 2):
                consider(_step(least, offset))
            refit = fit(best_codes)
            for candidate in (refit, _step(refit, -1), _step(refit, 1)):
                consider(candidate)
            scales[row, index] = best_scale
            codes[row, index] = best_codes.astype(np.int8)
    return codes, scales


def _weighted_error(weight, quantized, importance=None):
    rows, groups, group = quantized.codes.shape
    padded = torch.zeros((rows, groups * group), dtype=torch.float64)
    padded[:, : weight.shape[1]] = weight.double()
    channel = torch.zeros(groups * group, dtype=torch.float64)
    channel[: weight.shape[1]] = 1.0 if importance is None else importance.double()
    decoded = quantized.scales.double().unsqueeze(-1) * quantized.codes.double()
    residual = padded.reshape(rows, groups, group) - decoded
    return (channel.reshape(1, groups, group) * residual**2).sum(dim=2)


def _adversarial(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn((rows, columns), generator=generator) * 0.02
    weight[0] = 0.5  # flat row
    weight[1] *= 1e-7  # near-zero row
    weight[2, 5] = 0.9  # positive outlier
    weight[3, 70] = -0.8  # negative outlier
    return weight


@pytest.mark.parametrize("format", ["q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16"])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_search_matches_independent_reference(format, negative, weighted) -> None:
    weight = _adversarial(6, 200, 20261002)
    importance = (
        torch.rand(200, generator=torch.Generator().manual_seed(5)) * 3 if weighted else None
    )
    searched = search_quantize_matrix(
        weight, format, importance=importance, negative_scales=negative, device="cpu"
    )
    codes, scales = reference_search(weight, format, importance, negative)
    assert np.array_equal(searched.codes.numpy(), codes)
    assert np.array_equal(searched.scales.numpy().view(np.uint16), scales.view(np.uint16))


@pytest.mark.parametrize("format", ["q4_g64_fp16", "q5_g64_fp16", "q8_g32_fp16"])
def test_search_is_never_worse_than_round_to_nearest(format) -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn((48, 200), generator=generator) * 0.02
    weight[3, 5] = 0.4
    importance = torch.rand(200, generator=generator) * 3
    spec = get_format(format)
    rtn = quantize_matrix(weight, format, device="cpu")
    before = _weighted_error(weight, rtn, importance)
    for negative in (False, True):
        searched = search_quantize_matrix(
            weight, format, importance=importance, negative_scales=negative, device="cpu"
        )
        assert int(searched.codes.min()) >= spec.qmin
        assert int(searched.codes.max()) <= spec.qmax
        assert negative or bool((searched.scales.float() >= 0).all())
        after = _weighted_error(weight, searched, importance)
        assert bool((after <= before).all())
        assert float(after.sum()) < float(before.sum())


def test_search_uses_full_range_code_with_negative_scale() -> None:
    # Positive extreme 8.0: the full-range scale -1.0 maps it to code -8 exactly and every other
    # value is an exact multiple, so only the negative scale is error free.
    weight = torch.zeros((1, 64))
    weight[0, :4] = torch.tensor([8.0, -3.0, 5.0, 1.0])
    positive = search_quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    full = search_quantize_matrix(weight, "q4_g64_fp16", negative_scales=True, device="cpu")
    assert float(positive.scales[0, 0]) > 0
    assert float(full.scales[0, 0]) == -1.0
    assert full.codes[0, 0, :4].tolist() == [-8, 3, -5, -1]


def test_search_uses_positive_full_range_scale_without_negative_scales() -> None:
    # A negative extreme reaches code -8 under a positive scale, so it needs no signed scale.
    weight = torch.zeros((1, 64))
    weight[0, :4] = torch.tensor([-8.0, 3.0, -5.0, 1.0])
    searched = search_quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert float(searched.scales[0, 0]) == 1.0
    assert searched.codes[0, 0, :4].tolist() == [-8, 3, -5, 1]


def test_search_keeps_zero_and_underflow_groups_canonical() -> None:
    weight = torch.zeros((3, 64))
    weight[1, 0] = torch.finfo(torch.float32).tiny
    weight[2, 0] = 1.0
    searched = search_quantize_matrix(weight, "q6_g64_fp16", negative_scales=True, device="cpu")
    assert searched.scales.view(torch.int16)[0, 0] == 0
    assert torch.count_nonzero(searched.codes[0]) == 0
    # 2^-126 rounds to code 0 under every candidate, so the canonical 2^-24 scale stays.
    assert int(searched.scales.view(torch.int16)[1, 0]) == 1
    assert searched.codes[1, 0, 0].item() == 0
    assert float(searched.scales[2, 0]) != 0


def test_search_ignores_channels_without_importance() -> None:
    weight = torch.zeros((1, 64))
    weight[0, 0] = 1.0
    weight[0, 1:] = 0.013
    importance = torch.zeros(64)
    canonical = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    searched = search_quantize_matrix(weight, "q4_g64_fp16", importance=importance, device="cpu")
    assert torch.equal(searched.codes, canonical.codes)
    assert torch.equal(searched.scales, canonical.scales)


def test_search_rejects_invalid_inputs() -> None:
    weight = torch.zeros((1, 64))
    with pytest.raises(ValueError, match="one value per input channel"):
        search_quantize_matrix(weight, "q4_g64_fp16", importance=torch.ones(32), device="cpu")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        search_quantize_matrix(weight, "q4_g64_fp16", importance=-torch.ones(64), device="cpu")
    weight[0, 0] = float("nan")
    with pytest.raises(ValueError):
        search_quantize_matrix(weight, "q4_g64_fp16", device="cpu")


def test_signed_scales_roundtrip_through_row_split_codec() -> None:
    weight = _adversarial(6, 128, 11).to(torch.bfloat16)
    searched = search_quantize_matrix(weight, "q5_g64_fp16", negative_scales=True, device="cpu")
    assert bool((searched.scales < 0).any())
    payload = encode_row_split(searched.codes, searched.scales, "q5_g64_fp16", weight.shape)
    scales, codes = decode_row_split_codes(payload, "q5_g64_fp16", tuple(weight.shape))
    assert torch.equal(scales.view(torch.int16), searched.scales.view(torch.int16))
    assert torch.equal(codes, searched.codes)
    decoded = dequantize_row_split(
        payload, "q5_g64_fp16", tuple(weight.shape), dtype=torch.float32
    )
    expected = (searched.codes.float() * searched.scales.float().unsqueeze(-1)).reshape(6, 128)
    assert torch.equal(decoded, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_search_cuda_matches_cpu() -> None:
    weight = _adversarial(8, 256, 3)
    importance = torch.rand(256, generator=torch.Generator().manual_seed(9))
    cpu = search_quantize_matrix(
        weight, "q4_g64_fp16", importance=importance, negative_scales=True, device="cpu"
    )
    cuda = search_quantize_matrix(
        weight, "q4_g64_fp16", importance=importance, negative_scales=True, device="cuda"
    )
    assert torch.equal(cuda.codes.cpu(), cpu.codes)
    assert torch.equal(cuda.scales.cpu().view(torch.int16), cpu.scales.view(torch.int16))
