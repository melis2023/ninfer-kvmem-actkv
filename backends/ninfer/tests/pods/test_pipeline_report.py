from copy import deepcopy

import pytest

from scripts.pods.compare_pipeline import compare


def reports():
    single = {"schema": "flash-next-pipeline-1", "devices": [0], "artifact_id": "same-artifact",
              "source": {"source_sha256": "same-source"}, "drafts": 0, "kv": "int8_g64",
              "prefill_chunk": 512, "max_context": 131072, "exit_code": 0,
              "hardware": ["48 GB card"], "cases": []}
    for count in (4032, 33024, 131008):
        single["cases"].append({"prompt_tokens": count, "input_tokens": [7] * count,
            "needle_position": count // 2, "expected": "9517", "samples": [
                {"repeat": repeat, "pass": True, "output_tokens": [1] * 64, "content": "9517",
                 "reasoning": "", "finish_reason": 3, "reused_prompt_tokens": 0,
                 "prefill_seconds": time, "decode_seconds": time, "seconds": 2 * time}
                for repeat, time in enumerate((3.0, 1.0, 2.0))]})
    split = deepcopy(single)
    split["devices"] = [0, 1]
    split["hardware"] = ["24 GB card", "24 GB card"]
    return single, split


def test_report_keeps_all_samples_and_exposes_a_regression():
    single, split = reports()
    for sample in split["cases"][0]["samples"]:
        sample["decode_seconds"] *= 1.1
    result = compare(single, split)
    assert result["byte_identity"]
    timing = result["contexts"][0]["decode_seconds"]
    assert timing["single"] == [3, 1, 2]
    assert timing["single_median"] == 2
    assert timing["split_change_percent"] == pytest.approx(10)


def test_report_does_not_hide_a_cross_device_or_repeat_difference():
    single, split = reports()
    split["cases"][1]["samples"][2]["output_tokens"][0] = 9
    result = compare(single, split)
    assert not result["byte_identity"]
    assert not result["contexts"][1]["fixed_mode_equal"]
    assert result["contexts"][1]["matching_split_samples"] == 2


def test_post_eos_differences_are_accepted_and_remain_visible():
    single, split = reports()
    for report in (single, split):
        for case in report["cases"]:
            for sample in case["samples"]:
                sample["output_tokens"][4] = 248046
    for case in split["cases"]:
        for sample in case["samples"]:
            sample["output_tokens"][9:] = [9] * 55
            sample["content"] += " trailing continuation"
    result = compare(single, split, [248046])
    assert result["qualification_pass"]
    assert not result["byte_identity"]
    assert result["contexts"][0]["matching_through_eos_samples"] == 3
    assert result["contexts"][0]["different_tokens"] == [55] * 3
    assert not compare(single, split)["qualification_pass"]


@pytest.mark.parametrize("position", [0, 4])
def test_difference_before_or_at_eos_is_rejected(position):
    single, split = reports()
    for report in (single, split):
        for sample in report["cases"][0]["samples"]:
            sample["output_tokens"][4] = 248046
    for sample in split["cases"][0]["samples"]:
        sample["output_tokens"][position] = 9
    assert not compare(single, split, [248046])["qualification_pass"]


def test_post_eos_repeat_difference_is_still_rejected():
    single, split = reports()
    for report in (single, split):
        for sample in report["cases"][0]["samples"]:
            sample["output_tokens"][4] = 248046
    split["cases"][0]["samples"][2]["output_tokens"][9] = 9
    result = compare(single, split, [248046])
    assert result["contexts"][0]["matching_through_eos_samples"] == 3
    assert not result["qualification_pass"]


@pytest.mark.parametrize("change", ["artifact", "missing_repeat", "failed", "truncated", "infinite"])
def test_report_refuses_incomparable_or_incomplete_evidence(change):
    single, split = reports()
    if change == "artifact":
        split["artifact_id"] = "another-artifact"
    elif change == "missing_repeat":
        split["cases"][0]["samples"].pop()
    elif change == "failed":
        split["exit_code"] = 1
    elif change == "truncated":
        split["cases"][0]["input_tokens"].pop()
    else:
        split["cases"][0]["samples"][0]["decode_seconds"] = float("inf")
    with pytest.raises(ValueError):
        compare(single, split)
