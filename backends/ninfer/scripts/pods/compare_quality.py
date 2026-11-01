#!/usr/bin/env python3
"""Compare all streams and windows of two matching public Engine perplexity runs."""
import argparse
import json
import math
from pathlib import Path


def aggregate(value):
    count, total = value["scored_tokens"], value["total_nll"]
    mean, ppl = value["mean_nll"], value["perplexity"]
    if not isinstance(count, int) or count <= 0 or not all(map(math.isfinite, (total, mean, ppl))):
        raise ValueError("invalid score aggregate")
    if not math.isclose(mean, total / count, rel_tol=1e-10, abs_tol=1e-10) or \
            not math.isclose(ppl, math.exp(mean), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("score aggregate is inconsistent")


def difference(baseline, candidate):
    aggregate(baseline)
    aggregate(candidate)
    if baseline["scored_tokens"] != candidate["scored_tokens"]:
        raise ValueError("scored target counts differ")
    return {"scored_tokens": baseline["scored_tokens"],
            "baseline_ppl": baseline["perplexity"], "candidate_ppl": candidate["perplexity"],
            "mean_nll_delta": candidate["mean_nll"] - baseline["mean_nll"],
            "ppl_change_percent": (candidate["perplexity"] / baseline["perplexity"] - 1) * 100}


def compare(baseline, candidate):
    for field in ("schema_version", "metric", "corpus", "execution"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"quality reports differ in {field}")
    if baseline["schema_version"] != 4 or \
            baseline["artifact"]["architecture"] != candidate["artifact"]["architecture"]:
        raise ValueError("quality report schema or architecture differs")
    streams, windows = [], []
    for left, right in zip(baseline["streams"], candidate["streams"], strict=True):
        for field in ("id", "domain", "input_tokens", "unscored_tokens"):
            if left[field] != right[field]:
                raise ValueError(f"quality streams differ in {field}")
        streams.append({"id": left["id"], "domain": left["domain"], **difference(left, right)})
        for a, b in zip(left["windows"], right["windows"], strict=True):
            for field in ("index", "input_begin", "input_end", "target_begin", "target_end", "first_target"):
                if a[field] != b[field]:
                    raise ValueError(f"quality windows differ in {field}")
            if a["scored_tokens"] != a["target_end"] - a["target_begin"]:
                raise ValueError("quality window has an invalid target count")
            windows.append({"stream": left["id"], "index": a["index"], **difference(a, b)})
        for report in (left, right):
            if report["scored_tokens"] != sum(w["scored_tokens"] for w in report["windows"]) or \
                    not math.isclose(report["total_nll"], sum(w["total_nll"] for w in report["windows"]),
                                     rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("quality stream does not cover its windows")
    if not windows:
        raise ValueError("quality reports contain no windows")
    for report in (baseline, candidate):
        if len(report["streams"]) != report["corpus"]["stream_count"] or \
                report["overall"]["scored_tokens"] != sum(s["scored_tokens"] for s in report["streams"]) or \
                not math.isclose(report["overall"]["total_nll"],
                                 sum(s["total_nll"] for s in report["streams"]),
                                 rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("quality report does not cover its streams")
    times = [r["timing"]["score_seconds"] for r in (baseline, candidate)]
    if not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError("quality report timing is invalid")
    return {"baseline": baseline["artifact"], "candidate": candidate["artifact"],
            "corpus": baseline["corpus"], "execution": baseline["execution"],
            "overall": difference(baseline["overall"], candidate["overall"]),
            "streams": streams, "windows": windows,
            "worst_window": max(windows, key=lambda w: w["mean_nll_delta"]),
            "best_window": min(windows, key=lambda w: w["mean_nll_delta"]),
            "window_changes": {"higher_nll": sum(w["mean_nll_delta"] > 0 for w in windows),
                               "lower_nll": sum(w["mean_nll_delta"] < 0 for w in windows),
                               "equal_nll": sum(w["mean_nll_delta"] == 0 for w in windows)},
            "timing": {"baseline_seconds": times[0], "candidate_seconds": times[1],
                       "change_percent": (times[1] / times[0] - 1) * 100,
                       "scope": "causal scoring only; one sequential pair, no timing distribution"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.baseline.read_text()),
                             json.loads(args.candidate.read_text())), indent=2, allow_nan=False))
