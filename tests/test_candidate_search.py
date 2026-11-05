# test_candidate_search.py
#
# Offline tests for the candidate-search selectors and measurements and the pre-generation
# head sweep helpers, on synthetic pools. No model, dataset or network access.

import numpy as np
import torch

from experiments.approach1_latent_self_correction.analyze_candidate_pools import choose
from experiments.approach1_latent_self_correction.selection_metrics import (
    aurc,
    compute_curve,
    evaluate_pool,
    grouped_auroc,
    pool,
    select,
)
from experiments.approach1_latent_self_correction.sweep_pre_head import build, pass_rate_ranking, spearman


def candidate(answer, correct, head, probe=None, token=0.5):
    """One synthetic candidate."""
    return {"answer": answer, "correct": correct, "head_confidence": head,
            "probe_confidence": head if probe is None else probe, "token_confidence": token, "capped": False}


def record(greedy, samples, pre=0.5):
    """One synthetic pool record."""
    return {"dataset": "textvqa", "index": 0, "greedy": greedy, "samples": samples,
            "pre_generation": {"head": pre, "probe": pre}}


MIXED = record(candidate("cat", False, 0.4, token=0.9),
               [candidate("dog", True, 0.8, token=0.3), candidate("cat", False, 0.3), candidate("dog", True, 0.7)], pre=0.2)
EASY = record(candidate("7", True, 0.9), [candidate("7", True, 0.9), candidate("7", True, 0.9), candidate("8", False, 0.1)], pre=0.9)


def test_pool_prefix_includes_greedy_first():
    """Pool k is greedy then the first k samples."""
    assert [c["answer"] for c in pool(MIXED, 2)] == ["cat", "dog", "cat"]


def test_selectors():
    """Each selector picks the expected candidate, and ties keep greedy."""
    assert select(MIXED, 3, "greedy") == 0
    assert select(MIXED, 3, "head_best") == 1
    assert select(MIXED, 3, "token_best") == 0
    assert select(MIXED, 3, "oracle") == 1
    assert pool(MIXED, 3)[select(MIXED, 3, "majority_vote")]["answer"] == "dog"
    assert pool(MIXED, 3)[select(MIXED, 3, "head_vote")]["answer"] == "dog"
    assert select(MIXED, 3, "head_anchor_0.20") == 1 and select(MIXED, 3, "head_anchor_0.50") == 0
    assert select(EASY, 3, "head_best") == 0


def test_grouped_auroc_ignores_unanimous_pools():
    """Only mixed pools count; perfect within-question ranking gives 1."""
    unanimous = record(candidate("x", True, 0.1), [candidate("x", True, 0.2)])
    assert grouped_auroc([MIXED, unanimous], 3, "head") == 1.0
    assert grouped_auroc([MIXED], 3, "token") < 0.5


def test_evaluate_pool_headroom_and_mixed_accuracy():
    """Head selection recovers the whole oracle gap on these pools."""
    report = evaluate_pool([MIXED, EASY], 3)
    head = report["selectors"]["head_best"]
    assert report["pool"]["oracle_accuracy"] == 1.0 and report["selectors"]["greedy"]["accuracy"] == 0.5
    assert head["accuracy"] == 1.0 and head["headroom_captured"] == 1.0
    assert head["mixed_questions"] == 2 and head["mixed_accuracy"] == 1.0
    assert 0 < head["mixed_random_pick"] < 1


def test_aurc_rewards_good_ranking():
    """A score that puts the wrong answer last has lower AURC, and the best ranking has zero excess."""
    correct = [True, True, False, True]
    good = aurc(correct, [0.9, 0.8, 0.1, 0.7])
    bad = aurc(correct, [0.1, 0.8, 0.9, 0.7])
    assert good["aurc"] < bad["aurc"]
    assert abs(good["e_aurc"]) < 0.05


def test_compute_curve_routes_lowest_gate_first():
    """With a 50% budget, the low-gated mixed question is routed to the selector and fixed."""
    curve = compute_curve([MIXED, EASY], [MIXED["pre_generation"]["head"], EASY["pre_generation"]["head"]], 3, "head_best",
                          budgets=(0.5, 1.0))
    assert curve["0.5"]["gain"] == 0.5 and curve["1.0"]["gain"] == 0.5
    assert curve["0.5"]["generations_per_question"] == 2.5


def test_choose_skips_greedy_and_oracle():
    """The validation choice is the best real selector, preferring the smaller pool on a tie."""
    def entry(**accuracy):
        return {"pooled": {"selectors": {name: {"accuracy": value, "difference": value - 0.8} for name, value in accuracy.items()}}}
    validation = {"1": {"5": entry(greedy=0.8, head_best=0.82, oracle=0.9), "7": entry(greedy=0.8, head_best=0.82, oracle=0.95)},
                  "1.3": {"5": entry(greedy=0.8, head_best=0.81, oracle=0.97)}}
    choice = choose(validation)
    assert (choice["temperature"], choice["k"], choice["selector"]) == ("1", 5, "head_best")


def test_pre_head_helpers():
    """The head maps embeddings to one logit, the ranking loss prefers higher pass rates, Spearman handles ties."""
    head = build("mlp_256", 16, 0.0)
    assert head(torch.zeros(3, 16)).shape == (3, 1)
    rates = torch.tensor([0.9, 0.1])
    assert pass_rate_ranking(torch.tensor([2.0, -2.0]), rates) < pass_rate_ranking(torch.tensor([-2.0, 2.0]), rates)
    assert abs(spearman(np.array([1, 2, 2, 3]), np.array([1, 2, 2, 3])) - 1.0) < 1e-9


def _run_all() -> None:
    """Run this module without requiring pytest."""
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
