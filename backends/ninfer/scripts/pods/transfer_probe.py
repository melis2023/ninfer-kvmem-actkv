#!/usr/bin/env python3
"""Measure the pinned-host DMA cost of the CPU probe's real routed expert banks."""
import ctypes as c
import json
from pathlib import Path
import statistics
import struct

cuda = c.CDLL("/usr/local/cuda/lib64/libcudart.so")


def call(name, *args):
    rc = getattr(cuda, name)(*args)
    if rc:
        raise RuntimeError(f"{name}: CUDA error {rc}")


fixture = Path("/workspace/ninfer-work/models/cpu-q2-layer-12.bin").read_bytes()
h, width, experts = struct.unpack_from("<III", fixture)
assert (h, width, experts) == (2560, 640, 80)
projection_bytes = h * width // 64 * 18
host, device, stream, begin, end = [c.c_void_p() for _ in range(5)]
call("cudaSetDevice", c.c_int(0))
call("cudaHostAlloc", c.byref(host), c.c_size_t(len(fixture) - 12), c.c_uint(0))
call("cudaMalloc", c.byref(device), c.c_size_t(len(fixture) - 12))
call("cudaStreamCreate", c.byref(stream))
call("cudaEventCreate", c.byref(begin))
call("cudaEventCreate", c.byref(end))
c.memmove(host, fixture[12:], len(fixture) - 12)
try:
    for tokens in (1, 2, 5, 8):
        for reused in (True, False):
            selected = 10 if reused else 10 * tokens
            samples = []
            for repetition in range(10):
                call("cudaEventRecord", begin, stream)
                for projection in range(3):
                    for expert in range(selected):
                        offset = (projection * experts + expert) * projection_bytes
                        call("cudaMemcpyAsync", c.c_void_p(device.value + offset),
                             c.c_void_p(host.value + offset), c.c_size_t(projection_bytes),
                             c.c_int(1), stream)
                call("cudaEventRecord", end, stream)
                call("cudaEventSynchronize", end)
                elapsed = c.c_float()
                call("cudaEventElapsedTime", c.byref(elapsed), begin, end)
                if repetition:
                    samples.append(elapsed.value)
            print(json.dumps({"tokens": tokens, "reuse": reused, "unique_experts": selected,
                              "bytes": selected * projection_bytes * 3,
                              "median_ms": statistics.median(samples), "min_ms": min(samples),
                              "max_ms": max(samples), "scope": "all-miss H2D, no GPU compute"}),
                  flush=True)
finally:
    call("cudaEventDestroy", begin)
    call("cudaEventDestroy", end)
    call("cudaStreamDestroy", stream)
    call("cudaFree", device)
    call("cudaFreeHost", host)
