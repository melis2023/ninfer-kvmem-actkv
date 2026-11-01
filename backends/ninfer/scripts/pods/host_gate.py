#!/usr/bin/env python3
"""Check CUDA context, egress and uncached local storage before expensive work."""
import ctypes
from pathlib import Path
import subprocess
import time
import urllib.request

cuda = ctypes.CDLL("libcuda.so.1")
assert cuda.cuInit(0) == 0, "cuInit failed"
count = ctypes.c_int()
assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
with urllib.request.urlopen("https://huggingface.co", timeout=30) as response:
    assert response.status == 200
probe = Path("/workspace/ninfer-work/disk-probe")
start = time.monotonic()
try:
    subprocess.run(["dd", "if=/dev/zero", f"of={probe}", "bs=16M", "count=64",
                    "oflag=direct", "status=none"], check=True)
    bandwidth = 1024 / (time.monotonic() - start)
    print(f"disk_direct_write_mib_s={bandwidth:.1f}")
    assert bandwidth >= 700, "storage below 700 MiB/s"
finally:
    probe.unlink(missing_ok=True)
print(f"HOST_GATE_PASS devices={count.value}")
