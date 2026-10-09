# test_unified_q_head.py
#
# Offline tests for the unified Q-head and its sweep, run on small synthetic
# features instead of real Qwen3-VL states. The synthetic labels depend on
# both the prompt latent and the answer latent, so a head that reads both
# should beat a head that reads only one.
#
# Run with `python -m tests.test_unified_q_head`, or with pytest.

import torch

from experiments.approach1_latent_self_correction.sweep_unified_q_head import build_grid, run_sweep
from experiments.approach1_latent_self_correction.unified_q_head import (
    UnifiedConfig,
    UnifiedQHead,
    evaluate_modes,
    train_unified,
    unified_loss,
)

DEVICE = torch.device("cpu")


def make_split(count: int, seed: int, size: int = 24) -> dict:
    """Build fake prompt, answer, and reference features with labels tied to both halves."""
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.randn(count, size, generator=generator)
    answer = torch.randn(count, size, generator=generator)
    difficulty = prompt[:, 0] + 0.5 * prompt[:, 1]
    answer_signal = answer[:, 2]
    labels = ((difficulty + 1.5 * answer_signal + 0.3 * torch.randn(count, generator=generator)) > 0).float()
    reference = answer.clone()
    reference[:, 2] = 3.0
    return {
        "dataset": "synthetic",
        "labels": labels,
        "prompt_features": prompt,
        "generated_features": answer,
        "reference_features": reference,
    }


def test_pre_generation_mode_ignores_answer_half():
    """With no answer given, the output must not depend on any answer latent."""
    head = UnifiedQHead(24, "mlp_256").eval()
    prompt = torch.randn(5, 24)
    assert torch.allclose(head(prompt, None), head(prompt))
    assert not torch.allclose(head(prompt, torch.randn(5, 24)), head(prompt, None))


def test_loss_terms_present_only_when_enabled():
    """Each weighted term should show up in the loss breakdown only when its weight is non-zero."""
    split = make_split(64, 0)
    head = UnifiedQHead(24, "linear")
    args = (head, split["prompt_features"], split["generated_features"], split["labels"])
    _, parts = unified_loss(*args, UnifiedConfig(pre_weight=0.0))
    assert set(parts) == {"joint"}
    _, parts = unified_loss(
        *args,
        UnifiedConfig(pre_weight=1.0, rank_weight=0.1, pair_weight=0.25),
        reference=split["reference_features"],
    )
    assert set(parts) == {"joint", "pre", "rank", "pair"}


def test_training_beats_chance_in_both_modes():
    """A trained head should reach AUROC clearly above 0.5 in joint and pre-generation modes."""
    train, validation, test = make_split(1500, 1), make_split(400, 2), make_split(400, 3)
    best = train_unified(
        train, validation, UnifiedConfig(architecture="mlp_256", pair_weight=0.25), DEVICE, max_epochs=15
    )
    head = UnifiedQHead(24, "mlp_256")
    head.load_state_dict(best["state"])
    results = evaluate_modes(head, best["temperatures"], test, DEVICE)
    assert results["joint"]["correctness_auroc"] > 0.8
    assert results["pre_generation"]["correctness_auroc"] > 0.6
    assert results["joint"]["correctness_auroc"] > results["pre_generation"]["correctness_auroc"]


def test_pre_weight_helps_pre_generation_mode():
    """Training with the pre-generation term should not leave pre mode worse than training without it."""
    train, validation, test = make_split(1500, 4), make_split(400, 5), make_split(600, 6)
    scores = {}
    for pre_weight in (0.0, 1.0):
        best = train_unified(
            train, validation, UnifiedConfig(architecture="mlp_256", pre_weight=pre_weight), DEVICE,
            max_epochs=15, select="joint",
        )
        head = UnifiedQHead(24, "mlp_256")
        head.load_state_dict(best["state"])
        scores[pre_weight] = evaluate_modes(head, best["temperatures"], test, DEVICE)["pre_generation"]["correctness_auroc"]
    assert scores[1.0] >= scores[0.0] - 0.01


def test_quick_sweep_runs_end_to_end():
    """The sweep should return a winner, test summaries for both modes, and 3 test runs."""
    train, validation, test = make_split(600, 7), make_split(200, 8), make_split(200, 9)
    result, states = run_sweep(
        train, validation, test, build_grid(True, 0.0), DEVICE, max_epochs=3, patience=2, select="mean"
    )
    assert len(result["test_runs"]) == 3 and len(states) == 3
    assert set(result["test_summary"]) == {"joint", "pre_generation"}
    assert "mean" in result["test_summary"]["joint"]["correctness_auroc"]


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
