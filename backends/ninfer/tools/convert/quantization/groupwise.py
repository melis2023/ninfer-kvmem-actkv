"""Grouped symmetric quantization used by NInfer artifact converters.

The persistent numeric format fixes the code range, group size, and binary16
scale.  Model-specific recipes decide which tensors use those formats; this
module only performs the registered numeric transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.artifact.layouts import (
    row_split_geometry,
)
from tools.artifact.formats import QuantFormat, get_format

_FP16_MIN_SUBNORMAL = 2.0**-24


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


def _canonical_scale_words(
    max_abs: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical binary16 scales and binary32 reciprocals on the host.

    CUDA division is not correctly rounded at every binary16 scale boundary.
    The host oracle performs the specified division in binary64, explicitly
    rounds through binary32 and binary16, then computes the reciprocal in the
    same way.  A binary32 input divided by these small integer denominators has
    enough binary64 precision for the final binary32 rounding to be exact.
    """

    host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32)
        scale = raw_scale.astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")

    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values, including registered K padding.

    Scales are rounded to binary16 before codes are selected because those are
    the exact scales consumed after loading.  Padding values are zero and do
    not affect a partially populated final group.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if spec.name == "q2_g64_fp16":
        raise ValueError("q2_g64_fp16 requires encoded GSQ Q2_0 input, not grouped quantization")
    if weight.dim() != 2:
        raise ValueError(
            f"grouped quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    max_abs = grouped.abs().amax(dim=2)
    host_scales, host_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    scales = host_scales.to(target)
    reciprocal = host_reciprocal.to(target)
    codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales)


# Multipliers tried around the max-abs and full-range base scales: a dense sweep of the clipping
# region. It contains 1.0, so both base scales themselves are candidates.
SEARCH_RATIOS = tuple(round(0.70 + 0.02 * step, 2) for step in range(25))


def _round_scales(raw: np.ndarray, nonzero: np.ndarray) -> np.ndarray:
    """Round signed candidate scales to binary16 as ``_canonical_scale_words`` does.

    binary64 values round through binary32 to binary16. A nonzero group whose candidate
    underflows takes the smallest subnormal of the candidate's sign; overflow gives infinity,
    which the caller rejects.
    """

    with np.errstate(over="ignore", invalid="ignore"):
        scale = raw.astype(np.float64).astype(np.float32).astype(np.float16)
    underflow = (scale == 0) & nonzero & (raw != 0)
    if underflow.any():
        scale[underflow] = np.where(
            raw[underflow] < 0, -_FP16_MIN_SUBNORMAL, _FP16_MIN_SUBNORMAL
        ).astype(np.float16)
    return scale


def _ulp_neighbour(scale: np.ndarray, offset: int) -> np.ndarray:
    """The binary16 scale ``offset`` magnitude steps from ``scale``, keeping its sign.

    Steps that leave the finite nonzero range give zero, which the caller rejects."""

    words = scale.view(np.uint16)
    magnitude = (words & 0x7FFF).astype(np.int32) + offset
    legal = (words & 0x7FFF).astype(np.int32) > 0
    legal &= (magnitude > 0) & (magnitude <= 0x7BFF)
    shifted = (words & 0x8000) | np.clip(magnitude, 0, 0x7BFF).astype(np.uint16)
    return np.where(legal, shifted, 0).astype(np.uint16).view(np.float16)


def search_quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    importance: torch.Tensor | None = None,
    negative_scales: bool = False,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Choose each group's binary16 scale by minimizing weighted squared reconstruction error.

    The error of a group is ``sum_k importance[k] * (w[k] - scale * code[k])^2`` with codes
    selected exactly as the engine encoding assumes: ``round(w * binary32(1 / scale))`` clamped
    to the code interval. ``importance`` holds one finite nonnegative weight per input channel
    (length K), normally the mean squared activation from an importance matrix; without it every
    channel weighs the same. Candidates, in tie-break order:

    - the ``quantize_matrix`` result, which only a strictly smaller error replaces, so no group
      is worse than round-to-nearest;
    - ``SEARCH_RATIOS`` times the max-abs scale ``amax / qmax``;
    - ``SEARCH_RATIOS`` times the full-range scale ``extreme / qmin``, which maps the group's
      largest-magnitude value onto the most negative code. It is negative when that value is
      positive, and negative candidates compete only when ``negative_scales`` is set;
    - the max-abs scale times 1/4, 1/2, 2 and 4, and its +-1 and +-2 binary16 steps;
    - the weighted least-squares scale of the max-abs codes, times 1/2 and 2, and its +-1 and
      +-2 steps;
    - the weighted least-squares refit of the codes chosen so far, and its +-1 steps.

    Code selection is exact on every device. The errors are accumulated in binary64, so a CPU
    and a CUDA search agree except where two candidates' errors tie to binary64 rounding.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    # Validates rank, dtype and finiteness, and is the first candidate.
    baseline = quantize_matrix(weight, spec, device=device)
    target = baseline.codes.device
    geometry = row_split_geometry(spec, weight.shape)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical
    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    exact = grouped.to(torch.float64)

    if importance is None:
        weights = torch.ones(
            (1, geometry.groups_per_row, spec.group_size),
            dtype=torch.float64,
            device=target,
        )
    else:
        if importance.dim() != 1 or importance.numel() != geometry.k:
            raise ValueError(
                f"importance must have one value per input channel ({geometry.k})"
            )
        values = importance.detach().to(device=target, dtype=torch.float64)
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("importance must be finite and nonnegative")
        padded = torch.zeros(geometry.k_pad, dtype=torch.float64, device=target)
        padded[: geometry.k] = values
        weights = padded.reshape(1, geometry.groups_per_row, spec.group_size)

    def error(scales: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        residual = exact - scales.to(torch.float64).unsqueeze(-1) * codes.to(torch.float64)
        return (weights * residual * residual).sum(dim=2)

    def least_squares(codes: torch.Tensor) -> np.ndarray:
        codes = codes.to(torch.float64)
        numerator = (weights * exact * codes).sum(dim=2)
        denominator = (weights * codes * codes).sum(dim=2)
        fit = torch.where(
            denominator > 0, numerator / denominator.clamp_min(1e-300), 0.0
        )
        return fit.cpu().numpy()

    magnitude = grouped.abs()
    max_abs, position = magnitude.max(dim=2)
    extreme = grouped.gather(2, position.unsqueeze(-1)).squeeze(-1)
    nonzero = (max_abs > 0).cpu().numpy()
    host_max = max_abs.cpu().numpy().astype(np.float64)
    host_extreme = extreme.cpu().numpy().astype(np.float64)

    best_codes = baseline.codes.clone()
    best_scales = baseline.scales.clone()
    best_error = error(best_scales, best_codes)

    def consider(candidate: np.ndarray) -> None:
        nonlocal best_codes, best_scales, best_error
        valid = nonzero & np.isfinite(candidate) & (candidate != 0)
        if not negative_scales:
            valid &= candidate > 0
        if not valid.any():
            return
        scale = np.where(valid, candidate, np.float16(0))
        reciprocal = np.zeros(scale.shape, dtype=np.float32)
        reciprocal[valid] = (1.0 / scale[valid].astype(np.float64)).astype(np.float32)
        scales = torch.from_numpy(scale).to(target)
        codes = torch.clamp(
            torch.round(grouped * torch.from_numpy(reciprocal).to(target).unsqueeze(-1)),
            spec.qmin,
            spec.qmax,
        ).to(torch.int8)
        candidate_error = error(scales, codes)
        better = torch.from_numpy(valid).to(target) & (candidate_error < best_error)
        best_error = torch.where(better, candidate_error, best_error)
        best_scales = torch.where(better, scales, best_scales)
        best_codes = torch.where(better.unsqueeze(-1), codes, best_codes)

    for base in (host_max / float(spec.qmax), host_extreme / float(spec.qmin)):
        for ratio in SEARCH_RATIOS:
            consider(_round_scales(base * ratio, nonzero))
    canonical = baseline.scales.cpu().numpy()
    for exponent in (-2, -1, 1, 2):
        consider(_round_scales(canonical.astype(np.float64) * 2.0**exponent, nonzero))
    for offset in (-2, -1, 1, 2):
        consider(_ulp_neighbour(canonical, offset))
    fit = _round_scales(least_squares(baseline.codes), nonzero)
    consider(fit)
    for exponent in (-1, 1):
        consider(_round_scales(fit.astype(np.float64) * 2.0**exponent, nonzero))
    for offset in (-2, -1, 1, 2):
        consider(_ulp_neighbour(fit, offset))
    refit = _round_scales(least_squares(best_codes), nonzero)
    for candidate in (refit, _ulp_neighbour(refit, -1), _ulp_neighbour(refit, 1)):
        consider(candidate)
    return QuantizedMatrix(codes=best_codes, scales=best_scales)
