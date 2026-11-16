# Tests for the training-free baselines: voting rules and per-question costs.

from experiments.approach1_latent_self_correction.compute_baselines import methods, vote


def test_vote_normalizes_and_breaks_ties_by_order():
    assert vote(["Dog", "cat", "dog."]) == "Dog"
    assert vote(["a", "b"]) == "a"


def test_methods_costs_and_fallback():
    answers = ["cat", "dog", "cow"]
    q = {"key": ("docvqa", 0), "greedy": "cat", "samples": {"1": ["dog"] * 9},
         "corrections": {"cot": "dog", "cot_s0": "dog", "ground": "cow"}, "judge": {"cat": 0.1, "dog": 0.9, "cow": 0.2},
         "encoded": {a: {"correct": a == "dog", "tf_confidence": {"cat": 0.9, "dog": 0.5, "cow": 0.2}[a]} for a in answers}}
    m = methods(q, {"8b": {("docvqa", 0): {"correct": True}}})
    assert m["greedy"] == (False, 1.0, 1.0)
    assert m["self_consistency_5"][0] and m["self_consistency_5"][1] == 5.0
    assert m["reasoning_vote"][0] and m["set_judge_best"][0] and not m["set_token_best"][0]
    assert m["set_judge_best"][2] == 8.0 + 2 * 3 and m["model_8b"] == (True, 4.0, 4.0)
