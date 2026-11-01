#!/usr/bin/env python3
"""Compare every recorded MoE probe cell against the retained vector and GGUF baselines."""
import json
from pathlib import Path
import re
import sys


def read(path):
    log = path.read_text()
    if "GPU_PROBE_PASS" not in log or "GPU_PROBE_FAIL" in log:
        raise ValueError(f"probe did not pass: {path}")
    times = {}
    for t, reuse, cold, median, low, high in re.findall(
            r"gpu T=(\d+) reuse=(\d) cold=(\d) median_ms=(\S+) min_ms=(\S+) max_ms=(\S+)", log):
        times[int(t), int(reuse), int(cold)] = dict(zip(("median", "min", "max"),
                                                                     map(float, (median, low, high))))
    oracle = [float(v) for v in re.findall(r"relative_l2=(\S+)", log)]
    return times, {"min_relative_l2": min(oracle), "max_relative_l2": max(oracle),
                   "oracle_cases": len(oracle), "all_repeated": "repeated=0" not in log}


def main(root):
    routes = {name: read(root / f"{name}.log") for name in
              ("baseline-a16", "gguf", "native-a16", "native-a8")}
    keys = routes["baseline-a16"][0].keys()
    if not all(keys == data[0].keys() for data in routes.values()):
        raise ValueError("probe workloads differ")
    cells = []
    for t, reuse, cold in sorted(keys):
        times = {name: data[0][t, reuse, cold] for name, data in routes.items()}
        new = times["native-a16"]["median"]
        cells.append({"tokens": t, "reuse": bool(reuse), "after_memset": bool(cold),
                      "milliseconds": times,
                      "a16_change_vs_vector_pct": 100 * (new / times["baseline-a16"]["median"] - 1),
                      "a16_change_vs_gguf_pct": 100 * (new / times["gguf"]["median"] - 1)})
    print(json.dumps({"oracle": {name: data[1] for name, data in routes.items()}, "cells": cells}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
