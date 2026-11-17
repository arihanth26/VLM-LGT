# Tests for analyze_best_ranker.py: where a chosen answer comes from, and the abstention curve.

import numpy as np

from experiments.approach1_latent_self_correction.analyze_best_ranker import origin, selective


def test_origin_names_the_proposing_family():
    q = {"greedy": "a", "corrections": {"cot": "b", "esc_8b": "c", "cot_8b": "b", "ground": "d"}}
    sources = ("ground", "cot", "esc_8b", "cot_8b")
    assert origin(q, "a", sources) == "greedy kept"
    assert origin(q, "b", sources) == "2B and 8B both"
    assert origin(q, "c", sources) == "8B only"
    assert origin(q, "d", sources) == "2B corrections only"
    assert origin(q, "e", sources) == "2B sample"


def test_selective_ranks_by_confidence():
    correct = np.array([True, False, True, False])
    confidence = np.array([0.9, 0.1, 0.8, 0.2])
    s = selective(correct, confidence)
    assert s["accuracy_at_50"] == 1.0 and s["accuracy_at_100"] == 0.5
    # running accuracies 1, 1, 2/3, 1/2 -> errors 0, 0, 1/3, 1/2
    assert abs(s["aurc"] - (0 + 0 + 1 / 3 + 1 / 2) / 4) < 1e-9
