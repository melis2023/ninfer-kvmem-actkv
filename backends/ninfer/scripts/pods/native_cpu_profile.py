#!/usr/bin/env python3
"""Measure native Q2 import stages against direct source-plane extraction on real weights."""
import json
import statistics
import time

import numpy as np
import torch

from tools.convert.gguf_blocks import q2_native_source
from tools.convert.sources.gguf import GGUFFile

PATH = "/workspace/ninfer-work/models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
TENSOR = "blk.12.ffn_gate_exps.weight"


def timed(fn):
    values = []
    result = None
    for _ in range(3):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return result, statistics.median(values)


with GGUFFile(PATH) as gguf:
    for threads in (1, 4, 12):
        torch.set_num_threads(threads)
        for count in (640, 16384, 65536):
            source = q2_native_source(gguf, TENSOR, (count, 2560), lambda a, b: np.arange(a, b))
            words, read = timed(lambda: source.read_encoded(0, count))
            assert words.packed
            payload, pack = timed(lambda: words.codes.numpy().tobytes() + words.scales.numpy().tobytes())

            def extract():
                raw = gguf.read_blocks(TENSOR, 0, count).reshape(count, 40, 18)
                code = np.ascontiguousarray(raw[..., 2:]).tobytes()
                scale = np.ascontiguousarray(raw[..., :2]).tobytes()
                return code + scale  # these shapes need no plane alignment gap

            direct, extract_seconds = timed(extract)
            assert direct == payload
            print(json.dumps({"rows": count, "k": 2560, "threads": threads,
                              "source_seconds": read, "plane_bytes_seconds": pack,
                              "direct_planes_seconds": extract_seconds,
                              "bytes": len(payload)}), flush=True)
print("NATIVE_CPU_PROFILE_PASS", flush=True)
