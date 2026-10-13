# test_continual.py
#
# Offline tests for sequential (continual) training: the retention metrics against a
# hand-computed matrix, the replay sampler, and an end-to-end check on synthetic
# datasets whose labeling rules conflict, where plain sequential training must forget
# the first dataset and replay must forget less. No GPU or network.
#
# Run with `python -m tests.test_continual`, or with pytest.

import numpy as np
import torch

from experiments.approach1_latent_self_correction import continual, sweep_continual
from experiments.approach1_latent_self_correction.mixture_heads import HeadSpec

DEVICE = torch.device("cpu")
SIZE = 16


def make_dataset(count: int, seed: int, rule: str, index: int) -> dict:
    """A synthetic dataset whose correctness rule is a sign test on one prompt feature."""
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.randn(count, SIZE, generator=generator)
    answer = torch.randn(count, SIZE, generator=generator)
    if rule == "a":
        labels = (prompt[:, 0] > 0).float()
    elif rule == "b":
        labels = (prompt[:, 0] < 0).float()
    else:
        labels = (prompt[:, 1] > 0).float()
    return {
        "labels": labels,
        "prompt": prompt,
        "generated": answer,
        "prompt_corrupt": torch.randn(count, SIZE, generator=generator),
        "generated_corrupt": torch.randn(count, SIZE, generator=generator),
        "reference": answer.clone(),
        "dataset_id": torch.full((count,), index, dtype=torch.long),
    }


def make_world(rules=("a", "b", "c")) -> tuple[list, list, list]:
    """Train, validation, and test splits for datasets with the given rules."""
    train = [make_dataset(500, 10 + i, rule, i) for i, rule in enumerate(rules)]
    validation = [make_dataset(200, 20 + i, rule, i) for i, rule in enumerate(rules)]
    test = [make_dataset(300, 30 + i, rule, i) for i, rule in enumerate(rules)]
    return train, validation, test


def test_sequence_metrics_match_a_hand_computation():
    """Forgetting, backward transfer, final average, and retention on a known matrix."""
    matrix = np.array([[0.9, 0.5, 0.5], [0.8, 0.9, 0.5], [0.7, 0.85, 0.9]])
    result = continual.sequence_metrics(matrix, [0, 1, 2], np.array([0.9, 0.9, 0.9]))
    assert abs(result["final_avg"] - (0.7 + 0.85 + 0.9) / 3) < 1e-9
    assert abs(result["forgetting"] - 0.125) < 1e-9
    assert abs(result["bwt"] - (-0.125)) < 1e-9
    assert abs(result["retention"] - ((0.7 + 0.85 + 0.9) / 3 - 0.9)) < 1e-9
    # Forgetting is negative when a dataset scores higher at the end than at any earlier stage.
    other_order = continual.sequence_metrics(matrix, [2, 0, 1])
    assert abs(other_order["forgetting"] - (-0.15)) < 1e-9


def test_split_and_replay_sampler():
    """Datasets split by id, and the replay buffer holds the requested fraction of each earlier dataset."""
    train, _, _ = make_world()
    pooled = continual.concat(train)
    parts = continual.split_by_dataset(pooled, 3)
    assert [len(p["labels"]) for p in parts] == [500, 500, 500]
    buffer = continual.replay_sample(parts[:2], 0.1, seed=3)
    assert len(buffer["labels"]) == 100
    assert continual.replay_sample(parts[:2], 0.0, seed=3) is None
    assert continual.replay_sample([], 0.1, seed=3) is None


def test_plain_sequential_training_forgets_and_replay_forgets_less():
    """With conflicting rules, the first dataset is forgotten, and replay reduces that."""
    train, validation, test = make_world(("a", "b"))
    spec = HeadSpec("pre", "mlp_256", 1e-2, seed=1)
    plain = continual.run_sequence(spec, [0, 1], train, validation, test, DEVICE, replay=0.0, max_epochs=12, batch_size=128)
    replay = continual.run_sequence(spec, [0, 1], train, validation, test, DEVICE, replay=0.3, max_epochs=12, batch_size=128)
    plain_matrix = continual.matrix_of(plain, "main")
    replay_matrix = continual.matrix_of(replay, "main")
    assert plain_matrix[0, 0] > 0.9, "the first stage should learn the first dataset"
    assert plain_matrix[1, 0] < plain_matrix[0, 0] - 0.3, "plain sequential training should forget it"
    assert replay_matrix[1, 0] > plain_matrix[1, 0], "replay should remember more"


def test_variant_runner_reports_every_field():
    """The orchestrator returns matrices, metrics, an isolated reference, and a summary."""
    train, validation, test = make_world()
    names = ["x", "y", "z"]
    config = {**HeadSpec("pre", "linear", 1e-2).config()}
    result = sweep_continual.run_variant(
        "pre", config, {"forward": names}, train, validation, test, names, 0.0, DEVICE, 3, 2, 128
    )
    entry = result["orders"]["forward"]
    assert np.array(entry["matrix"]["main"]).shape == (3, 3)
    assert np.array(entry["grounding_matrix"]["main"]).shape == (3, 3)
    assert set(result["summary"]["main"]) == {"final_avg", "forgetting", "bwt", "retention"}
    assert set(result["isolated"]["main"]) == set(names)
    assert "mean" in entry["metrics"]["main"]["forgetting"]


def _run_all() -> None:
    """Run every test_* function in this module and report pass and fail counts."""
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
