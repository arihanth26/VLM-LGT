# test_prompted_retry.py
#
# Offline tests for low-confidence selection, retry prompting, and correction
# metrics. These tests do not load Qwen3-VL or access the network.

import numpy as np

from experiments.approach1_latent_self_correction.online_prompted_retry import add_policy_selections
from experiments.approach1_latent_self_correction.candidate_reranking import majority_index, select_indices
from experiments.approach1_latent_self_correction.deepen_selected_candidate import add_selections as add_deep_selections
from experiments.approach1_latent_self_correction.prompted_retry import (
    confidence_mask,
    retry_question,
    summarize,
)


def test_threshold_and_fraction_selection():
    """Threshold selection is strict and fractional selection keeps the lowest scores."""
    scores = np.array([0.8, 0.1, 0.5, 0.3])
    assert confidence_mask(scores, 0.5, None).tolist() == [False, True, False, True]
    assert confidence_mask(scores, 0.0, 0.5).tolist() == [False, True, False, True]


def test_retry_prompt_includes_evidence_instruction():
    """The retry sees the question and prior answer but is told to recheck the image."""
    prompt = retry_question("What value is shown?", "12")
    assert "What value is shown?" in prompt
    assert "previous answer was: 12" in prompt
    assert "visual evidence" in prompt
    assert "only the final answer" in prompt


def test_summary_counts_fixes_and_regressions():
    """Final policy accuracy replaces only successfully retried flagged answers."""
    records = [
        {"selected": True, "baseline_correct": False, "retry_correct": True},
        {"selected": True, "baseline_correct": True, "retry_correct": False},
        {"selected": False, "baseline_correct": True, "retry_correct": False},
        {"selected": True, "baseline_correct": False, "retry_error": "failed"},
    ]
    result = summarize(records, "selected")
    assert result["flagged"] == 3 and result["retried"] == 2
    assert result["fixed_errors"] == 1 and result["broken_answers"] == 1
    assert result["baseline_accuracy"] == result["final_accuracy"] == 0.5


def test_unselected_retry_does_not_replace_baseline():
    """A generated retry affects only policies that select its example."""
    records = [
        {"selected": False, "baseline_correct": True, "retry_correct": False},
        {"selected": False, "baseline_correct": False, "retry_correct": True},
    ]
    result = summarize(records, "selected")
    assert result["baseline_accuracy"] == result["final_accuracy"] == 0.5


def test_reranking_and_oracle_selections():
    """Reranking follows head scores while the oracle changes only fixable errors."""
    records = [
        {"head_confidence": 0.2, "retry_head_confidence": 0.7, "token_confidence": 0.9,
         "baseline_correct": False, "retry_correct": True},
        {"head_confidence": 0.8, "retry_head_confidence": 0.4, "token_confidence": 0.2,
         "baseline_correct": True, "retry_correct": False},
    ]
    add_policy_selections(records, threshold=0.5, fraction=0.5, random_seed=0)
    assert records[0]["head_rerank_selected"] and not records[1]["head_rerank_selected"]
    assert records[0]["head_gated_margin_0_1_selected"]
    assert records[0]["oracle_selected"] and not records[1]["oracle_selected"]


def test_candidate_selection_policies():
    """Candidate selectors use majority, token score, head score, and oracle as specified."""
    candidates = [
        {"answer": "A", "head_confidence": 0.2, "token_confidence": 0.9, "correct": False},
        {"answer": "B", "head_confidence": 0.8, "token_confidence": 0.4, "correct": True},
        {"answer": "B.", "head_confidence": 0.7, "token_confidence": 0.5, "correct": True},
    ]
    record = {"candidates": candidates}
    selected = select_indices(record)
    assert majority_index(candidates) == 1
    assert selected["token_best"] == 0
    assert selected["head_best"] == 1
    assert selected["oracle"] == 1


def test_deep_pass_selection_policies():
    """Deep-pass reranking and oracle replacement use the expected candidate."""
    records = [{
        "selected_candidate": {"head_confidence": 0.4, "correct": False},
        "deep_answer": {"head_confidence": 0.52, "correct": True},
    }]
    add_deep_selections(records)
    record = records[0]
    assert record["deep_head_rerank"] and record["deep_margin_0_1"]
    assert not record["deep_margin_0_2"] and record["deep_oracle"]


def _run_all() -> None:
    """Run this module without requiring pytest."""
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
