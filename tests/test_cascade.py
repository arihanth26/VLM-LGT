# Tests for the 2B -> 4B -> 8B cascade (analyze_cascade.py) and the scoring-pass option of gate_ranker_v3.py.

import numpy as np

from experiments.approach1_latent_self_correction.analyze_cascade import final, masks, setting


def toy() -> dict:
    """Four questions: 2B token confidence rises with the index; the 4B confidence is low on question 1."""
    a = {"dataset": np.array(["textvqa"] * 4), "c2": np.array([False, False, True, True]),
         "c4": np.array([True, False, True, False]), "c8": np.array([True, True, False, True]),
         "t4": np.array([0.9, 0.2, 0.8, 0.7]), "t8": np.array([0.95, 0.9, 0.1, 0.6])}
    for d in ("pre_head", "pre_probe", "after_head", "after_probe", "token"):
        a[d] = np.array([0.1, 0.2, 0.8, 0.9])
    return a


def test_cascade_masks_use_validation_thresholds():
    a = toy()
    to4, to8 = masks(a, a, "token", 0.5, 0.5, direct=False)
    assert to4.tolist() == [True, True, False, False]
    assert to8.tolist() == [False, True, False, False]  # the flagged question with the lower 4B confidence
    to4, to8 = masks(a, a, "token", 0.5, 0.0, direct=True)
    assert not to4.any() and to8.tolist() == [True, True, False, False]


def test_rules_and_cost():
    a = toy()
    to4, to8 = masks(a, a, "token", 0.5, 0.5, direct=False)
    assert final(a, to4, to8, "replace").tolist() == [True, True, True, True]
    # confident: question 1 has 2B 0.2, 4B 0.2, 8B 0.9 -> 8B; question 0 has 2B 0.1, 4B 0.9 -> 4B
    assert final(a, to4, to8, "confident").tolist() == [True, True, True, True]
    s = setting(a, a, "token", "replace", 0.5, 0.5, False)
    from experiments.approach1_latent_self_correction.analyze_escalation import FIX_COST  # in base-model passes
    assert abs(s["cost"] - (1 + FIX_COST["4b"] * 0.5 + FIX_COST["8b"] * 0.25)) < 1e-9 and s["to4"] == 0.5 and s["to8"] == 0.25


def test_confident_keeps_the_more_confident_2b_answer():
    a = toy()
    a["token"] = np.array([0.99, 0.2, 0.8, 0.9])  # question 0: the 2B answer beats the 4B one
    to4 = np.array([True, False, False, False])
    assert final(a, to4, np.zeros(4, dtype=bool), "confident")[0] == a["c2"][0]


def test_gating_scoring_passes_default_is_two():
    from experiments.approach1_latent_self_correction import gate_ranker_v3
    assert gate_ranker_v3.SCORING["passes"] == 2


def test_seconds_estimate_counts_each_component_once():
    from experiments.approach1_latent_self_correction import gate_ranker_v3 as g
    saved, passes = dict(g.LATENCY), g.SCORING["passes"]
    try:
        g.LATENCY.clear()
        g.LATENCY.update({"greedy": 0.1, "samples_10": 0.2, "grounding": 1.0, "reasoning": 1.5, "reasoning_x3": 1.8,
                          "greedy_8b": 0.15, "reasoning_8b": 2.0, "encode": 0.05, "judge": 0.04, "zoom": 0.1, "vcd": 0.3})
        sources = ("ground", "ground_ctx", "cot", "cot_s0", "cot_s1", "cot_s2", "esc_8b", "cot_8b")
        assert abs(g.generation_seconds(0, sources) - (0.1 + 1.0 + 1.5 + 1.8 + 0.15 + 2.0)) < 1e-9  # grounding once, the 3 samples once
        assert abs(g.generation_seconds(5, ()) - (0.1 + 0.1)) < 1e-9
        g.SCORING["passes"] = 1
        assert abs(g.scoring_seconds() - 0.05) < 1e-9
        g.SCORING["passes"] = 2
        assert abs(g.scoring_seconds() - 0.09) < 1e-9
        assert abs(g.base_seconds("token") - 0.1) < 1e-9 and abs(g.base_seconds("judge") - 0.14) < 1e-9
    finally:
        g.LATENCY.clear(); g.LATENCY.update(saved); g.SCORING["passes"] = passes


def test_evaluate_reports_seconds_when_timed():
    from experiments.approach1_latent_self_correction import gate_ranker_v3 as g
    saved = dict(g.LATENCY)
    try:
        g.LATENCY.clear(); g.LATENCY.update({"greedy": 0.1})
        table = {"greedy": np.array([True, False]), "scores": {"token": np.array([0.9, 0.1])},
                 "corrected": {"c": np.array([True, True])}, "passes": {"c": np.array([5.0, 5.0])}, "calls": {"c": 3.0},
                 "seconds": {"c": np.array([2.0, 2.0])}}
        e = g.evaluate(table, "c", "token", 0.5, table)
        assert e["share"] == 0.5 and abs(e["seconds"] - (0.1 + 2.0) / 2) < 1e-9
    finally:
        g.LATENCY.clear(); g.LATENCY.update(saved)
