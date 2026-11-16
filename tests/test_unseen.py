# Tests for the unseen-dataset transfer test: dataset normalization, split isolation, and that the
# configuration is chosen from in-domain validation only.

import json
import tempfile
from pathlib import Path

import numpy as np

from data.approach1_latent_self_correction.build_unseen_splits import ai2d_prompt, answer_letter
from experiments.approach1_latent_self_correction.correction_scale import ALL_DATASETS, DATASET_NAMES, SPLITS, UNSEEN_DATASETS


def test_unseen_datasets_come_after_the_in_domain_ones():
    # Seeds and orderings use positions in ALL_DATASETS; the in-domain positions must not move.
    assert ALL_DATASETS[: len(DATASET_NAMES)] == DATASET_NAMES and set(UNSEEN_DATASETS).isdisjoint(DATASET_NAMES)
    assert "unseen" in SPLITS and SPLITS[:3] == ("train", "val", "test")


def test_ai2d_answer_letter_from_index_letter_or_text():
    options = ["sun", "moon", "earth"]
    assert answer_letter("1", options) == "B" and answer_letter(2, options) == "C"
    assert answer_letter("a", options) == "A" and answer_letter("earth", options) == "C"


def test_ai2d_prompt_matches_scienceqa_format():
    prompt = ai2d_prompt("What eats grass?", ["cow", "lion"])
    assert prompt.splitlines() == ["Question: What eats grass?", "Options:", "A. cow", "B. lion", "Answer with the letter of the correct option only."]


def test_choice_uses_in_domain_validation_only():
    from experiments.approach1_latent_self_correction.evaluate_unseen import in_domain_choice
    row = lambda a: {"ranker_v3_scalars_nods": {"pooled": {"accuracy": a}}}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.json"
        path.write_text(json.dumps({"val": {"A": row(0.8), "B": row(0.9)}, "test": {"A": row(0.99), "B": row(0.1)}}))
        assert in_domain_choice(path) == "B"


def test_breakdown_skips_absent_datasets():
    from experiments.approach1_latent_self_correction.evaluate_unseen import breakdown
    datasets = np.array(["ai2d"] * 4)
    out = breakdown(datasets, np.array([True, False, False, True]), np.array([True, True, False, True]))
    assert set(out) == {"pooled", "ai2d"} and out["ai2d"]["fixed"] == 1


def test_chain_shards_partition_the_subset(monkeypatch=None):
    from experiments.approach1_latent_self_correction import correction_scale as cs
    original = cs.ranker_subset
    cs.ranker_subset = lambda dataset, split: list(range(0, 50, 5))
    try:
        shards = [cs.chain_indices("textvqa", "train", k, 3) for k in range(3)]
    finally:
        cs.ranker_subset = original
    assert sorted(i for s in shards for i in s) == list(range(0, 50, 5)) and not set(shards[0]) & set(shards[1])


def test_default_model_is_2b_with_heads():
    from experiments.approach1_latent_self_correction import correction_scale as cs
    assert cs.MODEL_SIZE == "2b" and cs.HEADS_AVAILABLE and cs.RESULTS_ROOT == cs.BASE_RESULTS and not cs.CHAIN
