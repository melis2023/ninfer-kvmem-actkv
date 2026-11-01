#!/usr/bin/env python3
"""Compare every recorded pipeline sample; report medians without hiding output differences."""
import argparse
import json
import math
from pathlib import Path
import statistics


def compare(single, split, eos_tokens=()):
    eos_tokens = frozenset(eos_tokens)
    if any(type(token) is not int or token < 0 for token in eos_tokens):
        raise ValueError("EOS token IDs must be nonnegative integers")

    def through_eos(tokens):
        return tokens[:next((i + 1 for i, token in enumerate(tokens) if token in eos_tokens), len(tokens))]

    for field in ("schema", "artifact_id", "source", "drafts", "kv", "prefill_chunk", "max_context"):
        if single[field] != split[field]:
            raise ValueError(f"pipeline reports differ in {field}")
    if single["devices"] != [0] or split["devices"] != [0, 1]:
        raise ValueError("expected one-card and two-card reports")
    expected = [4032, 33024, 131008]
    for report in (single, split):
        if report["schema"] != "flash-next-pipeline-1" or report["exit_code"] != 0:
            raise ValueError("pipeline qualification did not finish successfully")
        if [case["prompt_tokens"] for case in report["cases"]] != expected:
            raise ValueError("pipeline context coverage is incomplete")
        for case in report["cases"]:
            if len(case["input_tokens"]) != case["prompt_tokens"]:
                raise ValueError("pipeline prompt was truncated")
            samples = case["samples"]
            if len(samples) != 3 or [sample["repeat"] for sample in samples] != [0, 1, 2]:
                raise ValueError("every context needs exactly three repeats")
            if not all(s["pass"] and len(s["output_tokens"]) == 64 and
                       s["reused_prompt_tokens"] == 0 for s in samples):
                raise ValueError("an Engine sample failed or reused prompt state")
    rows = []
    equal = True
    qualified = True
    for one, two in zip(single["cases"], split["cases"], strict=True):
        for field in ("input_tokens", "needle_position", "expected"):
            if one[field] != two[field]:
                raise ValueError(f"pipeline inputs differ in {field}")
        fields = ("output_tokens", "content", "reasoning", "finish_reason")
        reference = one["samples"][0]
        repeats = [all(sample[field] == report["samples"][0][field] for field in fields)
                   for report in (one, two) for sample in report["samples"]]
        matching = [all(sample[field] == reference[field] for field in fields)
                    for sample in two["samples"]]
        visible_matching = [through_eos(sample["output_tokens"]) == through_eos(reference["output_tokens"])
                            for sample in two["samples"]]
        # Without a configured EOS in the reference, keep the original complete-output contract.
        if not eos_tokens.intersection(reference["output_tokens"]):
            visible_matching = matching
        equal = equal and all(repeats) and all(matching)
        qualified = qualified and all(repeats) and all(visible_matching)
        row = {"prompt_tokens": one["prompt_tokens"], "fixed_mode_equal": all(repeats),
               "matching_split_samples": sum(matching), "samples_per_device_count": 3,
               "matching_through_eos_samples": sum(visible_matching)}
        differences = [[index for index, (a, b) in enumerate(zip(reference["output_tokens"],
                         sample["output_tokens"], strict=True)) if a != b]
                       for sample in two["samples"]]
        row["different_tokens"] = [len(indices) for indices in differences]
        row["first_different_token"] = [indices[0] if indices else None for indices in differences]
        for metric in ("prefill_seconds", "decode_seconds", "seconds"):
            values = [[sample[metric] for sample in case["samples"]] for case in (one, two)]
            if not all(math.isfinite(value) and value > 0 for group in values for value in group):
                raise ValueError(f"missing or invalid {metric}")
            first, second = map(statistics.median, values)
            row[metric] = {"single": values[0], "split": values[1],
                           "single_median": first, "split_median": second,
                           "split_change_percent": (second / first - 1) * 100}
        row["prefill_tokens_per_second"] = [
            one["prompt_tokens"] / row["prefill_seconds"][key] for key in ("single_median", "split_median")]
        row["decode_tokens_per_second"] = [
            63 / row["decode_seconds"][key] for key in ("single_median", "split_median")]
        rows.append(row)
    return {"byte_identity": equal, "qualification_pass": qualified,
            "eos_token_ids": sorted(eos_tokens), "artifact_id": single["artifact_id"],
            "hardware": [single["hardware"], split["hardware"]], "contexts": rows,
            "measurement": "three requests per context; no discarded warmup; Engine cache disabled"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("single", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("--eos-token", type=int, action="append", default=[],
                        help="accept cross-host differences after this configured EOS token; repeat for multiple IDs")
    args = parser.parse_args()
    result = compare(json.loads(args.single.read_text()), json.loads(args.split.read_text()), args.eos_token)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["qualification_pass"] else 1)


if __name__ == "__main__":
    main()
