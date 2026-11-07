# test_correction_ranker.py
#
# Offline tests for the correction candidates (zoomed crops) and the within-question candidate
# ranker: candidate-set features, the listwise loss, and the untrained selectors. No model,
# dataset or network access.

import numpy as np
import torch
from PIL import Image

from experiments.approach1_latent_self_correction.correction_candidates import zoom_crops
from experiments.approach1_latent_self_correction.train_candidate_ranker import (
    SCALARS,
    Ranker,
    batch_tensors,
    candidate_set,
    listwise_loss,
    untrained_selectors,
)


def question():
    """One synthetic question with a greedy answer, samples at two temperatures and two corrections."""
    answers = ["cat", "dog", "Dog."]
    encoded = {
        "cat": {"tf_confidence": 0.9, "tokens": 2, "correct": False, "head_confidence": 0.7, "probe_confidence": 0.6},
        "dog": {"tf_confidence": 0.4, "tokens": 2, "correct": True, "head_confidence": 0.6, "probe_confidence": 0.5},
        "Dog.": {"tf_confidence": 0.3, "tokens": 3, "correct": True, "head_confidence": 0.65, "probe_confidence": 0.55},
    }
    return {"key": ("textvqa", 0), "answers": answers, "states": torch.randn(3, 2048).half(), "encoded": encoded,
            "question_state": torch.randn(2048).half(), "greedy": "cat",
            "samples": {"1": ["dog", "cat", "dog", "Dog."], "1.3": ["dog", "dog"]}, "corrections": {"vcd_1": "dog", "zoom_0": "cat"}}


def test_zoom_crops_cover_corners_and_upscale():
    """Four crops, each 60% of a side and upscaled 2x, capped at 2048 pixels."""
    crops = zoom_crops(Image.new("RGB", (1000, 500)))
    assert len(crops) == 4 and all(c.size == (1200, 600) for c in crops)
    assert max(zoom_crops(Image.new("RGB", (3000, 1000)))[0].size) == 2048


def test_candidate_set_features():
    """Distinct candidates in order, vote shares over normalized answers, greedy flag and correction share."""
    rows, features, labels = candidate_set(question(), "1", 3, ("vcd_1", "zoom_0"))
    assert rows == [0, 1] and labels.tolist() == [0.0, 1.0]
    assert features.shape == (2, len(SCALARS))
    assert abs(features[1, 0] - 2 / 3) < 1e-6 and features[0, 1] == 1.0 and features[1, 2] == 0.5
    rows, _, labels = candidate_set(question(), "1", 4, ())
    assert rows == [0, 1, 2] and labels.tolist() == [0.0, 1.0, 1.0]


def test_listwise_loss_prefers_right_candidates():
    """The loss is lower when a right candidate scores highest, and questions without a choice add nothing."""
    q = question()
    sets = [candidate_set(q, "1", 4, ())]
    batch = batch_tensors([q], sets, "cpu")
    good = listwise_loss(torch.tensor([[-2.0, 2.0, 1.0]]), batch, 0.0)
    bad = listwise_loss(torch.tensor([[2.0, -2.0, -1.0]]), batch, 0.0)
    assert good < bad
    only_right = {**q, "greedy": "dog", "samples": {"1": ["dog"]}, "corrections": {}}
    batch = batch_tensors([only_right], [candidate_set(only_right, "1", 1, ())], "cpu")
    assert float(listwise_loss(torch.tensor([[0.3]]), batch, 0.0)) == 0.0


def test_ranker_variants_score_every_candidate():
    """Each variant returns one score per padded candidate."""
    q = question()
    batch = batch_tensors([q, q], [candidate_set(q, "1", 4, ()), candidate_set(q, "1", 1, ())], "cpu")
    for variant in ("full", "scalars", "states"):
        scores = Ranker(variant, 32)(batch["question"], batch["answer"], batch["scalars"])
        assert scores.shape == (2, 3)
    assert batch["mask"].tolist() == [[True, True, True], [True, True, False]]


def test_untrained_selectors():
    """Greedy picks the greedy answer, head best the highest head score, vote the most common answer, oracle any right one."""
    result = untrained_selectors([question()], ("1", 4, ()))
    assert result == {"greedy": [False], "head_best": [False], "vote_then_head": [True], "oracle": [True]}


def _run_all() -> None:
    """Run this module without requiring pytest."""
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
