#!/usr/bin/env python3
"""Summarize every completed serving cell, preserving failed jobs and output differences."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def summarize(directory):
    job = {"directory": str(directory), "exit": int((directory / "exit").read_text()), "cells": []}
    for path in sorted(directory.glob("dma-*.json")):
        report = json.loads(path.read_text())
        samples = report["samples"]
        if [s["repeat"] for s in samples] != [0, 1, 2]:
            raise ValueError(f"{path}: incomplete repeated cell")
        outputs = [s["result"]["tokens"] for s in samples]
        if not all(len(out) == 64 and out == outputs[0] for out in outputs):
            raise ValueError(f"{path}: output count or fixed-mode repetition failed")
        timings = [s["result"]["timings"] for s in samples]
        prompt = timings[0]["prompt_n"]
        for value in timings:
            if value["cache_n"] != 0 or value["prompt_n"] != prompt or value["predicted_n"] != 64:
                raise ValueError(f"{path}: incomparable token counts or prefix reuse")
            if not all(math.isfinite(value[key]) and value[key] > 0
                       for key in ("prompt_ms", "predicted_ms", "predicted_per_second")):
                raise ValueError(f"{path}: invalid duration or rate")
            if not math.isclose(value["predicted_per_second"], 63000 / value["predicted_ms"]):
                raise ValueError(f"{path}: inconsistent decode rate")
        rates = [value["predicted_per_second"] for value in timings]
        hwm = re.search(r"^VmHWM:\s+(\d+) kB$", report["process_status"], re.M)
        if not hwm:
            raise ValueError(f"{path}: missing process peak RSS")
        io = []
        for sample in samples:
            before = sample["stats_before"].get("ngram_table", {})
            after = sample["stats_after"].get("ngram_table")
            if after is None:
                io.append(None)
                continue
            delta = {key: after[key] - before.get(key, 0) for key in
                     ("rows", "resident_rows", "batches", "read_seconds", "stalls", "stall_seconds")}
            if any(value < 0 for value in delta.values()):
                raise ValueError(f"{path}: a cumulative I/O counter decreased")
            delta["latency_us_since_startup"] = after["read_latency_us"]
            io.append(delta)
        job["cells"].append({"label": report["label"], "command": report["command"],
                             "prompt_tokens": prompt, "output_tokens": outputs[0],
                             "decode_tokens_per_second": rates,
                             "decode_median": statistics.median(rates),
                             "prompt_ms": [t["prompt_ms"] for t in timings],
                             "peak_rss_gib": int(hwm[1]) / 1024**2,
                             "cold_pages": report["cold_pages"], "io": io})
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path, nargs="+")
    args = parser.parse_args()
    reports = [summarize(path) for path in args.jobs]
    reference = next(cell for job in reports for cell in job["cells"]
                     if cell["label"] == "dma-1-k4-buffered")
    for job in reports:
        for cell in job["cells"]:
            cell["different_output_tokens"] = sum(a != b for a, b in
                zip(reference["output_tokens"], cell["output_tokens"], strict=True))
    print(json.dumps(reports, indent=2, allow_nan=False))
