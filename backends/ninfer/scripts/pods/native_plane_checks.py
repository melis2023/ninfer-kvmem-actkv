#!/usr/bin/env python3
"""Compare every converted native expert plane directly with retained source block bytes."""
from math import prod
from pathlib import Path

import numpy as np

from tools.artifact.reader import Artifact
from tools.convert.sources.gguf import GGUFFile

ROOT = Path("/workspace/ninfer-work/models")
MODEL = ROOT / "flash-next-native-q2-mtp.ninfer"
GGUF = ROOT / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
COMPANION = ROOT / "flash-next-q2_0-mtp.ninfer"


def region(artifact, name):
    binding = artifact.directory.bindings[name]
    if "object" in binding:
        obj = artifact.by_id[binding["object"]]
        return obj, 0, prod(obj.shape)
    assert len(binding["parts"]) == 1, name
    part = binding["parts"][0]
    return artifact.by_id[part["object"]], *part["range"]


def compare(artifact, source, name, tensor, rows, k, expert=0):
    obj, begin, end = region(artifact, name)
    assert end - begin == rows * k and begin % k == 0 and obj.shape[1] == k, name
    first = begin // k
    blocks = source.read_blocks(tensor, expert * rows, (expert + 1) * rows)
    kind = source.info(tensor).type_id
    if kind in (42, 8):
        assert obj.format == ("q2_g64_fp16" if kind == 42 else "q8_g32_fp16"), \
            name + " must preserve its trained native representation"
    if obj.format == "bf16":
        assert obj.layout == "contiguous_le_v1", name
        # This cross-check uses gguf-py's documented decoder, also used by conversion. It checks
        # framing, row selection and BF16 cast, and is not an independent source-format oracle.
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        import torch
        if kind == 42:
            raw = blocks.reshape(rows, k // 64, 18)
            scales = np.ascontiguousarray(raw[..., :2]).view("<f2").astype(np.float64)
            codes = ((raw[..., 2:, None].astype(np.int16) >>
                      (2 * np.arange(4, dtype=np.int16))) & 3) - 1
            decoded = (codes * scales[..., None]).reshape(rows, k).astype(np.float32)
        else:
            decoded = dequantize(np.ascontiguousarray(blocks), GGMLQuantizationType(kind))
        values = torch.from_numpy(np.array(decoded, copy=True)).to(torch.bfloat16)
        got = artifact.read_range(obj.offset + begin * 2, (end - begin) * 2)
        assert got == values.view(torch.int16).numpy().tobytes(), name + " BF16 cast"
        return
    group, block, code_row = (64, 18, k // 4) if kind == 42 else (32, 34, k)
    expected_format = "q2_g64_fp16" if kind == 42 else "q8_g32_fp16"
    assert kind in (42, 8) and obj.format == expected_format, name
    assert obj.layout == "row_split_k128_v1" and k % 128 == 0, name
    raw = blocks.reshape(rows, k // group, block)
    scale_offset = (obj.shape[0] * code_row + 255) // 256 * 256
    got_codes = artifact.read_range(obj.offset + first * code_row, rows * code_row)
    got_scales = artifact.read_range(obj.offset + scale_offset + first * (k // group) * 2,
                                     rows * (k // group) * 2)
    assert got_codes == np.ascontiguousarray(raw[..., 2:]).tobytes(), name + " codes"
    assert got_scales == np.ascontiguousarray(raw[..., :2]).tobytes(), name + " scales"


with Artifact(MODEL) as artifact, Artifact(COMPANION) as companion, GGUFFile(GGUF) as source:
    assert artifact.directory.components["ngram"] == companion.directory.components["ngram"]
    assert "ngram/table" not in artifact.directory.bindings
    experts = artifact.directory.components["text"]["config"]["num_experts"]
    checked = 0
    for layer in range(48):
        prefix = f"text/layers/{layer}/moe/"
        for expert in range(experts):
            for role, rows, k in (("gate", 640, 2560), ("up", 640, 2560),
                                 ("down", 2560, 640)):
                compare(artifact, source, prefix + f"experts/{expert}/{role}",
                        f"blk.{layer}.ffn_{role}_exps.weight", rows, k, expert)
                checked += 1
        for role, rows, k in (("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640)):
            compare(artifact, source, prefix + f"shared/{role}",
                    f"blk.{layer}.ffn_{role}_shexp.weight", rows, k)
            checked += 1
        print(f"NATIVE_PLANES layer={layer} checked={checked}", flush=True)
    print(f"NATIVE_PLANES_PASS projections={checked} bytes={artifact.file_bytes}", flush=True)
