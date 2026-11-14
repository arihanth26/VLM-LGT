# Tests for deployable gating (validation thresholds) and the v3 gating script's cost accounting.

import numpy as np

from experiments.approach1_latent_self_correction.gating import combine, gate, random_gate, threshold


def test_threshold_matches_budget_on_reference():
    reference = np.arange(100, dtype=float)
    for budget in (0.0, 0.2, 0.37, 1.0):
        assert gate(reference, threshold(reference, budget)).sum() == round(budget * 100)


def test_test_share_follows_scores_not_budget():
    # Thresholds are fixed on validation: if test scores are lower, more test questions are corrected.
    val = np.linspace(0, 1, 1000)
    limit = threshold(val, 0.2)
    assert gate(np.linspace(0, 1, 500), limit).mean() == 0.2
    assert gate(np.linspace(0, 0.5, 500), limit).mean() > 0.35


def test_each_decision_depends_only_on_its_score():
    limit = threshold(np.array([0.1, 0.5, 0.9]), 1 / 3)
    assert gate(np.array([0.1]), limit).tolist() == [True]
    assert gate(np.array([0.1, 0.0, 0.05]), limit).tolist() == [True, True, True]


def test_random_gate_and_combine():
    mask = random_gate(20000, 0.3)
    assert abs(mask.mean() - 0.3) < 0.02
    assert combine(np.array([True, False]), np.array([True, True]), np.array([False, False])).tolist() == [True, False]


def test_generation_calls_and_parse():
    from experiments.approach1_latent_self_correction.gate_ranker_v3 import generation_calls, parse
    from experiments.approach1_latent_self_correction.train_ranker_v3 import SETS
    assert parse("T1_k5_everything") == ("1", 5, "everything")
    assert parse("T1_k0_ground_cot") == ("1", 0, "ground_cot")
    assert generation_calls(0, SETS["cot"]) == 1
    assert generation_calls(0, SETS["ground_cot"]) == 4      # box + 2 grounded answers + reasoning
    assert generation_calls(5, SETS["everything"]) == 13     # samples 1 + vcd 2x2 + zoom 4 + box 1 + ground 2 + cot 1


def test_evaluate_gated_cost_and_outcome():
    from experiments.approach1_latent_self_correction.gate_ranker_v3 import evaluate
    table = {"greedy": np.array([True, False, False, True]), "dataset": np.array(["docvqa"] * 4),
             "scores": {"token": np.array([0.9, 0.1, 0.2, 0.8])},
             "corrected": {"c": np.array([False, True, True, True])}, "passes": {"c": np.array([10, 10, 10, 10])}, "calls": {"c": 5}}
    e = evaluate(table, "c", "token", 0.5, table)
    assert e["final"].tolist() == [True, True, True, True]
    assert e["share"] == 0.5 and e["passes"] == (1 + 1 + 10 + 10) / 4 and e["calls"] == (1 + 1 + 5 + 5) / 4
