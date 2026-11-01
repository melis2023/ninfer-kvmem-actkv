#!/usr/bin/env python3
"""Summarize the n-gram cache checks and the single CLI comparison without extrapolation."""
import json
from pathlib import Path
import re
import sys


def number(log, pattern):
    match = re.search(pattern, log)
    if not match:
        raise ValueError(f"missing metric: {pattern}")
    return float(match[1])


def seconds(value):
    value = value.strip()
    if value.endswith(" us"):
        return float(value[:-3]) / 1e6
    if value.endswith(" ms"):
        return float(value[:-3]) / 1e3
    if ":" in value:
        result = 0.0
        for part in value.split(":"):
            result = result * 60 + float(part)
        return result
    match = re.fullmatch(r"(?:(\d+)m )?([\d.]+)s", value)
    if not match:
        raise ValueError(f"invalid duration: {value}")
    return 60 * float(match[1] or 0) + float(match[2])


def cli_metrics(path):
    log = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
    out = {}
    for field, label in (("prefill_seconds", "text prefill"), ("decode_seconds", "decode"),
                         ("generation_seconds", "total")):
        out[field] = seconds(re.search(rf"^generate\s+{label}\s+(.+)$", log, re.M)[1])
    out.update(
        ngram_mean_us=number(log, r"ngram read per pass\s+([\d.]+) us"),
        ngram_stall_ms=number(log, r"ngram stalls\s+\d+ \(([\d.]+) ms\)"),
        ngram_hit_percent=number(log, r"ngram rows from RAM\s+([\d.]+)%"),
        prompt_tokens=int(number(log, r"prompt tokens\s+(\d+)")),
        generated_tokens=int(number(log, r"generated tokens\s+(\d+)")),
    )
    return out


def main(engine_root, root):
    if "NGRAM_COMPARISON_PASS" not in (root / "output.log").read_text():
        raise ValueError("the comparison did not complete")
    engine_log = (engine_root / "engine.log").read_text()
    if "NGRAM_CACHE_DONE failures=0" not in engine_log:
        raise ValueError("the Engine checks did not pass")
    engine = []
    for line in engine_log.splitlines():
        if line.startswith("NGRAM_CACHE "):
            engine.append({key: float(value) for key, value in
                           (entry.split("=") for entry in line.split()[1:])})
    if len(engine) != 12:
        raise ValueError("expected both widths, three modes and two repeats")
    for row in engine:
        if row["rows"] <= 0:
            raise ValueError("the Engine did not publish any row reads")
        if row["mode"] == 0 and row["hits"] != 0:
            raise ValueError("the disabled cache reported hits")
        if row["mode"] != 0 and row["repeat"] == 1 and row["hits"] != row["rows"]:
            raise ValueError("the repeated request still read rows from the file")
    samples = {mode: [cli_metrics(root / f"{mode}-1.err")]
               for mode in ("baseline", "cache")}
    reference = (root / "baseline-1.txt").read_bytes()
    for mode, runs in samples.items():
        for run, metrics in enumerate(runs, 1):
            if (root / f"{mode}-{run}.txt").read_bytes() != reference:
                raise ValueError(f"different answer: {mode}-{run}")
            for field in ("prompt_tokens", "generated_tokens"):
                if metrics[field] != samples["baseline"][0][field]:
                    raise ValueError(f"different workload: {mode}-{run} {field}")
    print(json.dumps({"engine": engine, "cli_samples": samples,
                      "limitation": "One CLI sample per mode; no stable throughput claim or per-mode peak RSS measurement."}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
