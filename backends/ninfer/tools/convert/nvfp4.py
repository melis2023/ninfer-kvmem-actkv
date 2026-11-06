"""Bounded weight-only NVFP4 round-to-nearest conversion for optional draft matrices."""
import struct
import torch


def quantize_rows(values, divisor):
    """Return packed E2M1 codes and natural E4M3 scale words; ties round to even."""
    values = values.to(torch.float32) * divisor
    if not torch.isfinite(values).all():
        raise ValueError('NVFP4 input must be finite')
    n, k = values.shape
    groups = values.reshape(n, k // 16, 16)
    scales = (groups.abs().amax(-1) / 6).to(torch.float8_e4m3fn)
    scales32 = scales.float()
    normalized = groups / torch.where(scales32 == 0, 1., scales32).unsqueeze(-1)
    magnitude = normalized.abs()
    midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=values.device)
    codes = torch.bucketize(magnitude.contiguous(), midpoints)
    codes += ((codes < 7) & (codes.remainder(2) == 1) &
              (magnitude == midpoints[codes.clamp(max=6)])).long()
    codes = (codes | (normalized.signbit().long() << 3)).to(torch.uint8).reshape(n, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.cpu(), scales.view(torch.uint8).cpu()


def nvfp4_maxabs(request):
    if request.target.format != 'nvfp4' or len(request.target.shape) != 2:
        raise ValueError('nvfp4_maxabs requires an NVFP4 matrix')
    n, k = request.target.shape
    if n % 128 or k % 64:
        raise ValueError('NVFP4 matrix requires N % 128 == 0 and K % 64 == 0')
    # ``policies`` carries the whole recipe's use table. Only this matrix's inputs
    # decide whether the weight-only route applies; other components may legitimately
    # keep A8/A4 policies.
    if any(
        request.policies[key] != 'A16Only'
        for item in request.inputs
        for key in item.uses
    ):
        raise ValueError('nvfp4_maxabs is weight-only; use A16Only')
    chunk = max(128, request.rows_per_chunk // 128 * 128)
    maximum = 0.
    for begin in range(0, n, chunk):
        values = request.values(begin*k, min(n, begin+chunk)*k).float()
        if not torch.isfinite(values).all():
            raise ValueError('NVFP4 input must be finite')
        maximum = max(maximum, values.abs().max().item())
    divisor = struct.unpack('<f', struct.pack('<f', 2688. / maximum if maximum else 1.))[0]
    if not (0 < divisor < float('inf')):
        raise ValueError('NVFP4 weight divisor is not representable in FP32')
    word = struct.pack('<f', divisor)

    def produce(output):
        for begin in range(0, n, chunk):
            end = min(n, begin+chunk)
            values = request.values(begin*k, end*k).reshape(end-begin, k)
            codes, scales = quantize_rows(values, divisor)
            output.write_codes(begin, codes, scales, word)
    return request.job(produce=produce)
