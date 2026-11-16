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
    from experiments.approach1_latent_self_correction.gate_ranker_v3 import parse
    from experiments.approach1_latent_self_correction.train_ranker_v3 import MULTI_SETS, SETS, generation_calls
    assert parse("T1_k5_everything") == ("1", 5, "everything")
    assert parse("T1_k0_ground_cot_multi") == ("1", 0, "ground_cot_multi")
    assert generation_calls(0, SETS["cot"]) == 2                   # greedy + reasoning
    assert generation_calls(0, SETS["ground_cot"]) == 5            # greedy + box + 2 grounded + reasoning
    assert generation_calls(0, MULTI_SETS["ground_cot_multi"]) == 8  # + 3 sampled reasoning answers


def test_reproduction_check_counts_questions():
    from experiments.approach1_latent_self_correction import gate_ranker_v3 as g
    import json, tempfile
    from pathlib import Path
    tables = {s: {"greedy": np.zeros(1000, bool), "corrected": {"c": np.r_[np.ones(500, bool), np.zeros(500, bool)]}} for s in ("val", "test")}
    row = lambda a: {"ranker_v3_scalars_ds": {"pooled": {"accuracy": a}}}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.json"
        path.write_text(json.dumps({"val": {"c": row(0.502)}, "test": {"c": row(0.5)}}))
        assert g.reproduction(tables, path, "scalars_ds")["c"]["match"]          # 2 questions apart: allowed
        path.write_text(json.dumps({"val": {"c": row(0.51)}, "test": {"c": row(0.5)}}))
        assert not g.reproduction(tables, path, "scalars_ds")["c"]["match"]      # 10 apart: stop


def test_evaluate_gated_cost_and_outcome():
    from experiments.approach1_latent_self_correction.gate_ranker_v3 import evaluate
    table = {"greedy": np.array([True, False, False, True]), "dataset": np.array(["docvqa"] * 4),
             "scores": {"token": np.array([0.9, 0.1, 0.2, 0.8])},
             "corrected": {"c": np.array([False, True, True, True])}, "passes": {"c": np.array([10, 10, 10, 10])}, "calls": {"c": 5}}
    e = evaluate(table, "c", "token", 0.5, table)
    assert e["final"].tolist() == [True, True, True, True]
    assert e["share"] == 0.5 and e["passes"] == (1 + 1 + 10 + 10) / 4 and e["calls"] == (1 + 1 + 5 + 5) / 4
