#!/usr/bin/env python3
"""Compare full-vocabulary logits at identical teacher-forced positions, without a quality claim."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    report = json.loads(path.read_text())
    if report["dtype"] != "bf16-le" or report["fixed_mode_repeats"] is not True:
        raise ValueError("expected repeated BF16 logits")
    vocab, domain = report["vocab"], report["domain"]
    if not 0 < domain <= vocab or not report["records"]:
        raise ValueError("invalid logit geometry")
    payload = path.parent / report["payload"]
    if payload.parent != path.parent or payload.stat().st_size != 2 * vocab * len(report["records"]):
        raise ValueError("invalid logit payload")
    for index, record in enumerate(report["records"]):
        if record["offset"] != index * vocab * 2:
            raise ValueError("logit records are not contiguous")
    words = np.fromfile(payload, dtype="<u2").astype("<u4") << 16
    values = words.view("<f4").reshape(-1, vocab)[:, :domain].astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite public logits")
    return report, values


def compare(baseline_path, candidate_path):
    baseline, a = read(baseline_path)
    candidate, b = read(candidate_path)
    for field in ("dtype", "vocab", "domain", "prefill_chunk", "kv", "prefixes", "records"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"teacher-forced reports differ in {field}")
    rows = []
    for record, x, y in zip(baseline["records"], a, b, strict=True):
        delta = y - x
        first, second = int(np.argmax(x)), int(np.argmax(y))
        norm, peak = float(np.linalg.norm(x)), float(np.max(np.abs(x)))
        if norm == 0 or peak == 0:
            raise ValueError("zero reference logits cannot define a relative error")
        top = np.partition(x, -2)[-2:]
        p = x - np.max(x)
        q = y - np.max(y)
        p -= np.log(np.exp(p).sum())
        q -= np.log(np.exp(q).sum())
        rows.append({"prefix": record["prefix"], "step": record["step"],
                     "relative_l2": float(np.linalg.norm(delta)) / norm,
                     "max_error_over_peak": float(np.max(np.abs(delta))) / peak,
                     "mean_absolute_error": float(np.mean(np.abs(delta))),
                     "kl_baseline_candidate": float(np.sum(np.exp(p) * (p - q))),
                     "top1_equal": first == second, "baseline_top1": first,
                     "candidate_top1": second, "baseline_top1_margin": float(top[1] - top[0]),
                     "candidate_choice_baseline_gap": float(x[first] - x[second])})
    return {"baseline": str(baseline_path), "candidate": str(candidate_path),
            "scope": "identical teacher-forced inputs; resident reference is not an independent oracle",
            "execution": [{k: r[k] for k in ("residency", "devices", "dma_share", "kv", "prefill_chunk")}
                          for r in (baseline, candidate)],
            "records": rows, "top1_equal": sum(r["top1_equal"] for r in rows),
            "record_count": len(rows), "exact_equal": bool(np.array_equal(a, b)),
            "max_relative_l2": max(r["relative_l2"] for r in rows),
            "max_kl": max(r["kl_baseline_candidate"] for r in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([compare(args.baseline, path) for path in args.candidates],
                     indent=2, allow_nan=False))
