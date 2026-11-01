"""Native Prism PTQ1_0 base-3 planes over groups of 128 ternary coordinates."""
from __future__ import annotations

import torch


def pack(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if codes.ndim != 2 or codes.shape[1] != 128:
        raise ValueError("PTQ1 codes must have shape [groups,128]")
    if bool(((codes < -1) | (codes > 1)).any()):
        raise ValueError("PTQ1 codes must be in {-1,0,1}")
    digits = codes.to(torch.int32) + 1
    planes = []
    for start, width, stages in ((0, 16, 5), (80, 8, 5), (120, 2, 4)):
        values = digits[:, start:start + width * stages].reshape(-1, stages, width)
        powers = torch.tensor([3 ** i for i in reversed(range(stages))],
                              device=codes.device, dtype=torch.int32).view(1, stages, 1)
        integer = (values * powers).sum(1)
        denominator = 3 ** stages
        planes.append(((integer * 256 + denominator - 1) // denominator).to(torch.uint8))
    return torch.cat(planes[:2], dim=1), planes[2]


def unpack(base: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    base = base.reshape(-1, 24).to(torch.int32)
    high = high.reshape(-1, 2).to(torch.int32)
    decoded = []
    for words, stages in ((base[:, :16], 5), (base[:, 16:], 5), (high, 4)):
        decoded.append(torch.stack([(((words * 3 ** i) & 255) * 3 // 256) - 1
                                    for i in range(stages)], dim=1).flatten(1))
    return torch.cat(decoded, dim=1).to(torch.int8)
