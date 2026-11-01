import copy
import math

import pytest

from scripts.pods.compare_quality import compare


def aggregate(means):
    total = sum(means)
    return {"scored_tokens": len(means), "total_nll": total,
            "mean_nll": total / len(means), "perplexity": math.exp(total / len(means))}


def report(first=1.0, second=2.0):
    windows = [{"index": i, "input_begin": 2 * i, "input_end": 2 * i + 3,
                "target_begin": 2 * i + 1, "target_end": 2 * i + 3, "first_target": 1,
                **aggregate([mean, mean])} for i, mean in enumerate((first, second))]
    overall = aggregate([first, first, second, second])
    return {"schema_version": 4, "metric": "causal", "corpus": {"stream_count": 1},
            "execution": {"context_tokens": 3}, "artifact": {"architecture": "qwen4_exp"},
            "streams": [{"id": "fixture", "domain": "text", "input_tokens": 5,
                         "unscored_tokens": 1, "windows": windows, **overall}],
            "overall": overall, "timing": {"score_seconds": 2.0}}


def test_keeps_small_regression_and_improvement_in_the_same_run():
    result = compare(report(), report(1.002, 1.999))
    assert result["overall"]["mean_nll_delta"] == pytest.approx(0.0005)
    assert result["window_changes"] == {"higher_nll": 1, "lower_nll": 1, "equal_nll": 0}
    assert result["worst_window"]["index"] == 0
    assert result["best_window"]["index"] == 1
    assert len(result["windows"]) == 2


def test_refuses_equal_counts_scored_from_different_history():
    candidate = report()
    candidate["streams"][0]["windows"][1]["input_begin"] = 1
    with pytest.raises(ValueError, match="input_begin"):
        compare(report(), candidate)


def test_refuses_a_missing_window_even_if_the_overall_aggregate_was_retained():
    candidate = copy.deepcopy(report())
    candidate["streams"][0]["windows"].pop()
    with pytest.raises(ValueError):
        compare(report(), candidate)


def test_refuses_changed_execution_settings():
    candidate = report()
    candidate["execution"]["context_tokens"] = 4
    with pytest.raises(ValueError, match="execution"):
        compare(report(), candidate)
