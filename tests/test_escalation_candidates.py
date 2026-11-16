# Tests for the escalation candidates in ranker v3 (enabled by --with-escalation). The flag changes module
# state, so every test restores it.

import copy

import numpy as np
import torch

from experiments.approach1_latent_self_correction import train_ranker_v3 as v3
from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES


def _with_escalation(check):
    saved = (copy.deepcopy(v3.GROUPS), dict(v3.SETS), v3.SCALARS_V3)
    try:
        v3.SETS.update(v3.MULTI_SETS)
        v3.enable_escalation()
        check()
    finally:
        v3.GROUPS.clear(); v3.GROUPS.update(saved[0])
        v3.SETS.clear(); v3.SETS.update(saved[1])
        v3.SCALARS_V3 = saved[2]


def test_disabled_by_default_so_old_models_load():
    assert "escalate" not in v3.GROUPS and "esc8b" not in v3.SETS
    assert v3.RankerV3("scalars_ds", 8).network[0].in_features == len(v3.SCALARS_V3)


def test_enable_adds_one_agreement_feature_before_the_dataset_columns():
    before = len(v3.SCALARS_V3)
    def check():
        assert len(v3.SCALARS_V3) == before + 1
        assert v3.SCALARS_V3[-len(DATASET_NAMES) - 1] == "from_escalate"
        assert v3.SCALARS_V3[-len(DATASET_NAMES):] == tuple(f"dataset_{d}" for d in DATASET_NAMES)
        assert v3.RankerV3("scalars_nods", 8).network[0].in_features == len(v3.SCALARS_V3) - len(DATASET_NAMES)
        assert set(v3.SETS["ground_cot_multi_esc"]) >= {"cot_s0", "esc_8b", "up2"}
    _with_escalation(check)
    assert len(v3.SCALARS_V3) == before


def test_candidate_features_include_8b_agreement():
    def check():
        answers = ["cat", "dog"]
        q = {"key": ("docvqa", 0), "answers": answers, "greedy": "cat", "samples": {"1": []},
             "corrections": {"cot": "dog", "esc_8b": "dog"}, "judge": {a: 0.5 for a in answers},
             "encoded": {a: {"tf_confidence": 0.5, "tokens": 1, "correct": a == "dog", "head_confidence": 0.5, "probe_confidence": 0.5} for a in answers}}
        rows, features, labels = v3.candidate_set(q, "1", 0, v3.SETS["ground_cot_multi_esc8b"])
        column = list(v3.SCALARS_V3).index("from_escalate")
        assert features[rows.index(1), column] == 1.0 and features[rows.index(0), column] == 0.0
        assert labels.tolist() == [0.0, 1.0]
    _with_escalation(check)


def test_8b_costs_four_calls():
    assert v3.generation_calls(0, ("esc_8b",)) == 5.0
    assert v3.generation_calls(0, ("cot",)) == 2


def test_drop_zeroes_the_ablated_columns_only():
    answers = ["cat", "dog"]
    q = {"key": ("docvqa", 0), "answers": answers, "greedy": "cat", "samples": {"1": ["dog"]},
         "corrections": {"cot": "dog"}, "judge": {"cat": 0.9, "dog": 0.2},
         "encoded": {a: {"tf_confidence": 0.5, "tokens": 1, "correct": a == "dog", "head_confidence": 0.7, "probe_confidence": 0.6} for a in answers}}
    full = v3.candidate_set(q, "1", 1, ("cot",))[1]
    try:
        v3.DROPPED.update(v3.FEATURE_GROUPS["judge"])
        dropped = v3.candidate_set(q, "1", 1, ("cot",))[1]
    finally:
        v3.DROPPED.clear()
    judge = [i for i, n in enumerate(v3.SCALARS_V3) if n in ("judge", "judge_logit")]
    other = [i for i in range(len(v3.SCALARS_V3)) if i not in judge]
    assert (dropped[:, judge] == 0).all() and (full[:, judge] != 0).any()
    assert np.array_equal(dropped[:, other], full[:, other])
