"""Exact T5 layout from JGamboa/ninfer-4090-windows (Apache-2.0, see NOTICE).

Two 13-byte units of 64 ternary coordinates per 128-column scale group.
This codec rearranges represented integers; it never requantizes weights.
"""
from __future__ import annotations

import torch


def _columns(device):
    return torch.tensor([[20 * (i // 4) + 4 * m + i % 4 for m in range(5)]
                         for i in range(12)] + [[60, 61, 62, 63, 64]],
                        dtype=torch.long, device=device)


def pack(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.int8 or codes.ndim != 2 or codes.shape[1] != 128:
        raise ValueError("T5 codes must be int8 [groups,128]")
    if bool(((codes < -1) | (codes > 1)).any()):
        raise ValueError("T5 codes must be in {-1,0,1}")
    units = codes.reshape(-1, 64).to(torch.int32) + 1
    units = torch.cat((units, torch.zeros_like(units[:, :1])), dim=1)
    digits = units[:, _columns(codes.device)]
    powers = torch.tensor([81, 27, 9, 3, 1], device=codes.device)
    integers = (digits * powers).sum(-1)
    return ((256 * integers + 242) // 243).to(torch.uint8).reshape(-1, 26)


def unpack(base: torch.Tensor) -> torch.Tensor:
    if base.dtype != torch.uint8 or base.numel() % 26:
        raise ValueError("T5 codes require whole 26-byte scale groups")
    words = base.reshape(-1, 13).to(torch.int32)
    digits = torch.stack([(((words * 3**i) & 255) * 3) >> 8 for i in range(5)], -1)
    powers = torch.tensor([81, 27, 9, 3, 1], device=base.device)
    canonical = (256 * (digits * powers).sum(-1) + 242) // 243
    if bool((canonical != words).any()) or bool((digits[:, 12, 4] != 0).any()):
        raise ValueError("T5 requires canonical bytes and a zero padding trit")
    output = torch.empty((words.shape[0], 65), dtype=torch.int32, device=base.device)
    output[:, _columns(base.device).flatten()] = digits.flatten(1)
    return (output[:, :64] - 1).to(torch.int8).reshape(-1, 128)
