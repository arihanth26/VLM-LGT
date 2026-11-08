# test_correction_v2.py
#
# Offline tests for the second-round correction helpers (box parsing and expansion, answer extraction
# from step-by-step reasoning) and the second ranker's candidate features. No model or network access.

import numpy as np
import torch

from experiments.approach1_latent_self_correction.correction_v2 import expand_box, extract_final, parse_box
from experiments.approach1_latent_self_correction.train_ranker_v2 import SCALARS_V2, RankerV2, batch_tensors, candidate_set


def test_parse_box_reads_the_relative_grid_and_rejects_bad_replies():
    """Numbers on the 0-1000 grid scale to pixels; too few numbers or an empty box give None."""
    assert parse_box('{"bbox_2d": [100, 200, 500, 600]}', 2000, 1000) == (200, 200, 1000, 600)
    assert parse_box("[1200, 50, 1800, 400]", 2000, 1000) == (1200, 50, 1800, 400)
    assert parse_box("no box here", 100, 100) is None
    assert parse_box("[10, 10, 10, 10]", 100, 100) is None
    assert parse_box('```json\n{"bbox_2d": [18, 105, 984, 844], "label": "chart 2"}\n```', 1000, 1000) == (18, 105, 984, 844)


def test_expand_box_adds_margin_and_minimum_size():
    """A tiny box grows to the minimum size and stays inside the image."""
    left, top, right, bottom = expand_box((50, 50, 52, 52), 1000, 500)
    assert right - left >= 199 and bottom - top >= 99
    assert left >= 0 and top >= 0 and right <= 1000 and bottom <= 500


def test_extract_final_answer():
    """The text after the last 'Answer:', else the last line."""
    assert extract_final("The bar is 12.\nSo the total is 30.\nAnswer: 30") == "30"
    assert extract_final("answer: 4\nWait, recount.\nAnswer: **5**") == "5"
    assert extract_final("First line\nlast line") is None
    assert extract_final("Step 1: read the axis. Step 2: the bar") is None


def test_v2_features_include_the_judge():
    """Candidate features append the judge score and its logit to the first ranker's nine scalars."""
    q = {"key": ("textvqa", 0), "answers": ["cat", "dog"], "states": torch.randn(2, 2048).half(), "question_state": torch.randn(2048).half(),
         "greedy": "cat", "samples": {"1": ["dog", "dog"]}, "corrections": {"ground": "dog"}, "judge": {"cat": 0.2, "dog": 0.9},
         "encoded": {a: {"tf_confidence": 0.5, "tokens": 2, "correct": a == "dog", "head_confidence": 0.6, "probe_confidence": 0.6} for a in ("cat", "dog")}}
    rows, features, labels = candidate_set(q, "1", 2, ("ground",))
    assert features.shape == (2, len(SCALARS_V2)) and abs(features[1, 9] - 0.9) < 1e-6 and labels.tolist() == [0.0, 1.0]
    batch = batch_tensors([q], [(rows, features, labels)], "cpu")
    for variant in ("full", "full_judge", "scalars_judge"):
        assert RankerV2(variant, 16)(batch["question"], batch["answer"], batch["scalars"]).shape == (1, 2)


def _run_all() -> None:
    """Run this module without requiring pytest."""
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
