# test_correction_stats.py
#
# Offline tests for the correction statistics: Wilson intervals, the exact sign test, the paired
# bootstrap, AUROC with ties, and the per-policy outcome helpers on small synthetic records.
#
# Run with `python -m tests.test_correction_stats`, or with pytest.

import math

from experiments.approach1_latent_self_correction import correction_stats as stats


def test_wilson_and_mcnemar_known_values():
    """The interval brackets the proportion, and the sign test matches hand-computed p-values."""
    low, high = stats.wilson(81, 100)
    assert 0.72 < low < 0.81 < high < 0.88
    assert stats.wilson(0, 0) == (0.0, 0.0)
    assert math.isclose(stats.mcnemar_p(4, 1), 0.375)   # 2 * P(X <= 1), n = 5
    assert stats.mcnemar_p(0, 0) == 1.0
    assert math.isclose(stats.mcnemar_p(10, 0), 2 / 1024)


def test_paired_difference_counts_and_sign():
    """Fixed and broken answers are counted per question and the difference has the right sign."""
    baseline = [True, True, False, False, True, False]
    other = [True, False, True, True, True, False]
    result = stats.paired_difference(baseline, other, boot=500)
    assert (result["fixed"], result["broken"]) == (2, 1)
    assert math.isclose(result["difference"], 1 / 6)
    assert result["ci_low"] <= result["difference"] <= result["ci_high"]


def test_auroc_handles_ties_and_perfect_ranking():
    """AUROC is 1 for a perfect score, 0 for a reversed one, and 0.5 for a constant score."""
    assert stats.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert stats.auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert stats.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_policy_outcomes_follow_the_selection_flags():
    """A policy keeps the baseline where its flag is off and takes the second answer where it is on."""
    records = [
        {"baseline_correct": True, "retry_correct": False, "retry_all_selected": True, "never_selected": False},
        {"baseline_correct": False, "retry_correct": True, "retry_all_selected": True, "never_selected": False},
    ]
    base, final = stats.retry_outcomes(records, "retry_all_selected")
    assert base == [True, False] and final == [False, True]
    assert stats.retry_outcomes(records, "never_selected")[1] == [True, False]
    deep = [{"selected_candidate": {"correct": True}, "deep_answer": {"correct": False, "answer": "A candidate answer was: x", "token_probabilities": [0.5] * 10}, "deep_replace_all": True}]
    assert stats.deep_outcomes(deep, "deep_replace_all") == ([True], [False])
    failures = stats.format_failures(deep)
    assert failures["echoed"] == 1 and failures["broken_echoed"] == 1 and failures["capped"] == 0


def _run_all() -> None:
    """Run every test_* function in this module and report pass and fail counts."""
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
