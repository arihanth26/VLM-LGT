# test_ranker_v3.py
#
# Offline tests for the third ranker's candidate features (source agreement, dataset one-hot) and the
# per-dataset router. No model or network access.

import numpy as np
import torch

from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.train_ranker_v2 import SCALARS_V2
from experiments.approach1_latent_self_correction.train_ranker_v3 import (
    GROUPS, SCALARS_V3, SETS, RankerV3, batch_tensors, candidate_set, router,
)


def _question(dataset="docvqa", index=0):
    answers = ["cat", "dog", "cow"]
    return {"key": (dataset, index), "answers": answers, "states": torch.randn(3, 2048).half(), "question_state": torch.randn(2048).half(),
            "greedy": "cat", "samples": {"1": ["dog", "cat"]},
            "corrections": {"zoom_0": "dog", "zoom_1": "dog", "zoom_2": "cow", "zoom_3": "dog", "cot": "cow", "ground": "cat"},
            "judge": {a: 0.5 for a in answers},
            "encoded": {a: {"tf_confidence": 0.5, "tokens": 2, "correct": a == "cow", "head_confidence": 0.6, "probe_confidence": 0.6} for a in answers}}


def test_sets_include_reasoning_plus_crops():
    assert set(SETS["zoom_cot"]) == {"zoom_0", "zoom_1", "zoom_2", "zoom_3", "cot"}


def test_v3_features_add_source_agreement_and_dataset():
    q = _question()
    rows, features, labels = candidate_set(q, "1", 0, SETS["zoom_cot"])
    assert features.shape == (3, len(SCALARS_V3))
    extra = {name: features[:, len(SCALARS_V2) + i] for i, name in enumerate(SCALARS_V3[len(SCALARS_V2):])}
    by_answer = {q["answers"][r]: i for i, r in enumerate(rows)}
    assert abs(extra["from_zoom"][by_answer["dog"]] - 0.75) < 1e-6 and abs(extra["from_zoom"][by_answer["cow"]] - 0.25) < 1e-6
    assert extra["from_cot"][by_answer["cow"]] == 1.0 and extra["from_cot"][by_answer["dog"]] == 0.0
    assert np.all(extra["from_ground"] == 0.0)  # ground not offered in this set
    assert np.all(extra["dataset_docvqa"] == 1.0) and np.all(extra["dataset_textvqa"] == 0.0)
    assert labels[by_answer["cow"]] == 1.0


def test_v3_ranker_shapes():
    q = _question()
    s = candidate_set(q, "1", 2, SETS["zoom_ground_cot"])
    batch = batch_tensors([q], [s], "cpu")
    for variant in ("full_ds", "scalars_ds"):
        assert RankerV3(variant, 16)(batch["question"], batch["answer"], batch["scalars"]).shape == (1, len(s[0]))


def test_router_picks_per_dataset_on_validation():
    qs = {s: [{"key": (d, i)} for d in DATASET_NAMES[:2] for i in range(2)] for s in ("val", "test")}
    out = {"val": {("A", "x"): [True, True, False, False], ("B", "y"): [False, False, True, True]},
           "test": {("A", "x"): [True, False, False, False], ("B", "y"): [False, False, True, False]}}
    choices, routed = router(qs, out)
    assert choices[DATASET_NAMES[0]]["configuration"] == "A" and choices[DATASET_NAMES[1]]["configuration"] == "B"
    assert routed["test"] == [True, False, True, False]


def _run_all() -> None:
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
