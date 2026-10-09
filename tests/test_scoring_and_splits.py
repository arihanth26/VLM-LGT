# test_scoring_and_splits.py
#
# Offline tests for the Q-head dataset helpers: the per-dataset answer
# scoring in scoring.py and the group-aware split logic in
# build_qhead_splits.py. No network, model, or datasets library needed.
#
# Run with `python -m tests.test_scoring_and_splits`, or with pytest.

from data.approach1_latent_self_correction.build_qhead_splits import (
    most_common_answer,
    scienceqa_prompt,
    split_by_group,
)
from experiments.approach1_latent_self_correction.scoring import (
    anls_similarity,
    levenshtein,
    normalize_vqa_answer,
    parse_option_letter,
    score_with_metric,
    vqa_accuracy,
)


def test_normalize_vqa_answer():
    """Articles, punctuation, case, and number words should not change a match."""
    assert normalize_vqa_answer("The Nokia!") == "nokia"
    assert normalize_vqa_answer("Two") == "2"
    assert normalize_vqa_answer("1,000") == "1000"
    assert normalize_vqa_answer("3.5") == "3.5"


def test_vqa_accuracy_matches_official_values():
    """One agreeing annotator scores 0.3, two score 0.6, three or more score 0.9 or higher."""
    ten = ["nokia"] * 1 + ["other"] * 9
    assert abs(vqa_accuracy("nokia", ten) - 0.3) < 1e-9
    assert abs(vqa_accuracy("nokia", ["nokia"] * 2 + ["x"] * 8) - 0.6) < 1e-9
    assert abs(vqa_accuracy("nokia", ["nokia"] * 3 + ["x"] * 7) - 0.9) < 1e-9
    assert vqa_accuracy("nokia", ["nokia"] * 10) == 1.0
    assert vqa_accuracy("", []) == 0.0


def test_vqa_binary_label_needs_two_annotators():
    """The correct label turns on at two agreeing annotators."""
    one, _ = score_with_metric("vqa_accuracy", ["a"] + ["b"] * 9, "a")
    two, _ = score_with_metric("vqa_accuracy", ["a", "a"] + ["b"] * 8, "a")
    assert not one and two


def test_levenshtein_and_anls():
    """Edit distance and ANLS similarity behave on simple strings."""
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert anls_similarity("University of California", ["university of california"]) == 1.0
    assert anls_similarity("1/9/93", ["1/8/93"]) < 0.9
    correct, _ = score_with_metric("anls", ["P. Carter", "p. carter"], "p. carter")
    assert correct


def test_option_letter_parsing():
    """Several reply styles should resolve to the same option letter."""
    assert parse_option_letter("B") == "B"
    assert parse_option_letter("(c)") == "C"
    assert parse_option_letter("D. Paris") == "D"
    assert parse_option_letter("Answer: B") == "B"
    assert parse_option_letter("The answer is c") == "C"
    assert parse_option_letter("Paris") is None
    assert score_with_metric("mcq_letter", ["B"], "b.")[0]
    assert not score_with_metric("mcq_letter", ["B"], "A")[0]


def test_split_by_group_never_shares_a_group():
    """No group id may land in two splits, and sizes should be met."""
    groups = [f"g{i // 4}" for i in range(400)]
    result = split_by_group(groups, {"val": 50, "test": 100}, seed=0)
    assert len(result["val"]) == 50 and len(result["test"]) == 100
    val_groups = {groups[i] for i in result["val"]}
    test_groups = {groups[i] for i in result["test"]}
    assert not val_groups & test_groups


def test_split_by_group_is_deterministic():
    """The same seed must give the same split."""
    groups = [f"g{i % 37}" for i in range(500)]
    assert split_by_group(groups, {"val": 40, "test": 80}, 7) == split_by_group(groups, {"val": 40, "test": 80}, 7)


def test_prompt_and_reference_helpers():
    """ScienceQA prompts list lettered options, and the common answer ignores blanks."""
    prompt = scienceqa_prompt(
        {"hint": "", "question": "Which is hotter?", "choices": ["Sun", "Moon"]}
    )
    assert "A. Sun" in prompt and "B. Moon" in prompt and "Context" not in prompt
    assert most_common_answer(["", "a", "b", "a"]) == "a"
    assert most_common_answer(["", ""]) == ""


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
