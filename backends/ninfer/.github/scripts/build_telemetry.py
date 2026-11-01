"""Record build resource use and compiler-cache counters without credentials."""
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def available_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total",
                 "virtual_available", "extended_available")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available
    elif Path("/proc/meminfo").is_file():
        match = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        if match:
            return int(match[1]) * 1024
    return None


def cache_stats():
    result = subprocess.run(["ccache", "--print-stats"], capture_output=True,
                            text=True, check=True, timeout=5)
    return {key: int(value) for key, value in
            (line.split() for line in result.stdout.splitlines())}


def sample(destination):
    report = {"cpu_count": os.cpu_count(), "available_memory_bytes": available_memory(),
              "free_disk_bytes": shutil.disk_usage(Path.cwd()).free}
    try:
        report.update(cache_stats())
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        report["cache_statistics_unavailable"] = type(error).__name__
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as output:
        output.write(json.dumps(report) + "\n")
    print("Build resources and cache: " + json.dumps(report), flush=True)
