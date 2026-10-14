# test_mixture.py
#
# Offline tests for the mixture and new-loss code: the image corruptions, the
# head losses (including a check that the old loss is reproduced exactly when the
# new terms are off), the grounding metrics, the mixture builder, and a quick
# end-to-end sweep. Everything runs on small synthetic features, no GPU or network.
#
# Run with `python -m tests.test_mixture`, or with pytest.

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.approach1_latent_self_correction import build_mixture, sweep_mixture
from experiments.approach1_latent_self_correction.corruption import corrupt_image, shuffle_patches, swap_partners
from experiments.approach1_latent_self_correction.mixture_heads import (
    HeadSpec,
    build_head,
    grounding_metrics,
    head_loss,
    to_device,
    train_head,
)
from experiments.approach1_latent_self_correction.unified_q_head import UnifiedConfig, UnifiedQHead, unified_loss

DEVICE = torch.device("cpu")
SIZE = 16


def make_split(count: int, seed: int, with_reference: bool = True, datasets: int = 2) -> dict:
    """Synthetic mixture split: labels depend on both states, corrupted states are noise."""
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.randn(count, SIZE, generator=generator)
    answer = torch.randn(count, SIZE, generator=generator)
    labels = ((prompt[:, 0] + answer[:, 1] + 0.3 * torch.randn(count, generator=generator)) > 0).float()
    data = {
        "labels": labels,
        "prompt_features": prompt,
        "generated_features": answer,
        "prompt_features_corrupt": torch.randn(count, SIZE, generator=generator),
        "generated_features_corrupt": torch.randn(count, SIZE, generator=generator),
        "dataset_id": torch.arange(count) % datasets,
    }
    if with_reference:
        reference = answer.clone()
        reference[:, 1] = 3.0
        data["reference_features"] = reference
    return data


def tensors_of(data: dict) -> dict:
    """Move a synthetic split into the tensor layout the trainer expects."""
    tensors = to_device(data, DEVICE)
    tensors["dataset_id"] = data["dataset_id"]
    return tensors


def test_swap_partners_has_no_fixed_points():
    """Every example must be paired with a different example."""
    for count in (2, 3, 50, 501):
        partners = swap_partners(count, seed=7)
        assert sorted(partners) == list(range(count))
        assert all(partner != index for index, partner in enumerate(partners))


def test_shuffle_patches_keeps_pixels_but_changes_layout():
    """Patch shuffling is a permutation of pixels, and it changes the image."""
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    shuffled = shuffle_patches(image, seed=3)
    assert shuffled.size == image.size
    assert sorted(np.array(shuffled).reshape(-1, 3).tolist()) == sorted(np.array(image).reshape(-1, 3).tolist())
    assert np.any(np.array(shuffled) != np.array(image))
    partner = Image.new("RGB", (10, 10), "red")
    assert corrupt_image(image, "swap", 0, partner).size == (10, 10)


def test_noise_and_mask_corruptions():
    """Noise keeps the size and destroys detail, mask blanks the requested share, both are deterministic."""
    from experiments.approach1_latent_self_correction.corruption import add_noise, mask_patches

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
    noisy = add_noise(image, seed=1, alpha=0.15)
    assert noisy.size == image.size and noisy.tobytes() == add_noise(image, seed=1, alpha=0.15).tobytes()
    clean_corr = np.corrcoef(np.array(image).ravel(), np.array(noisy).ravel())[0, 1]
    mild_corr = np.corrcoef(np.array(image).ravel(), np.array(add_noise(image, 1, 0.9)).ravel())[0, 1]
    assert clean_corr < mild_corr, "a smaller alpha should keep less of the image"
    masked = np.array(mask_patches(image, seed=2, fraction=0.5, grid=8))
    gray = (masked == 128).all(axis=2).mean()
    assert 0.45 < gray < 0.55, gray
    assert corrupt_image(image, "noise", 3).size == image.size and corrupt_image(image, "mask", 3, strength=0.25).size == image.size


def test_old_loss_is_reproduced_exactly():
    """With ground off, the mixture loss must equal the existing unified loss term for term."""
    data = make_split(64, 1)
    batch = {k: v.float() for k, v in to_device(data, DEVICE).items()}
    spec = HeadSpec("unified", "linear", 1e-3, pre_weight=1.0, pair_weight=0.25)
    head = build_head(spec, SIZE).eval()
    new_total, _ = head_loss(head, spec, batch)
    old_total, _ = unified_loss(
        head, batch["prompt"], batch["generated"], batch["labels"],
        UnifiedConfig(pre_weight=1.0, pair_weight=0.25), batch["reference"],
    )
    assert abs(float(new_total) - float(old_total)) < 1e-6


def test_ground_term_appears_only_when_enabled():
    """The grounding term is in the loss breakdown only when its weight is positive."""
    data = make_split(64, 2)
    batch = {k: v.float() for k, v in to_device(data, DEVICE).items()}
    for kind in ("pre", "verifier", "unified"):
        off = HeadSpec(kind, "linear", 1e-3)
        on = HeadSpec(kind, "linear", 1e-3, ground_weight=0.5)
        assert "ground" not in head_loss(build_head(off, SIZE), off, batch)[1]
        assert "ground" in head_loss(build_head(on, SIZE), on, batch)[1]


def test_grounding_metrics_values():
    """Order accuracy is 1 when clean always beats corrupt, and the mean drop is reported."""
    clean = np.array([0.9, 0.8, 0.7, 0.6])
    corrupt = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([1.0, 1.0, 0.0, 1.0])
    result = grounding_metrics(clean, corrupt, labels)
    assert result["all"]["order_accuracy"] == 1.0
    assert result["correct"]["count"] == 3
    assert abs(result["all"]["mean_drop"] - 0.5) < 1e-9
    tie = grounding_metrics(clean, clean, labels)
    assert tie["all"]["order_accuracy"] == 0.5


def test_grounding_loss_teaches_a_gap():
    """Training with the grounding term should score clean inputs above corrupted ones."""
    train, validation, test = (tensors_of(make_split(n, s)) for n, s in ((600, 3), (200, 4), (200, 5)))
    names = ["a", "b"]
    plain = train_head(HeadSpec("verifier", "linear", 1e-2), train, validation, test, names, DEVICE, max_epochs=8, batch_size=128)
    grounded = train_head(HeadSpec("verifier", "linear", 1e-2, ground_weight=1.0), train, validation, test, names, DEVICE, max_epochs=8, batch_size=128)
    gap_plain = plain["test"]["grounding"]["main"]["pooled"]["correct"]["mean_drop"]
    gap_grounded = grounded["test"]["grounding"]["main"]["pooled"]["correct"]["mean_drop"]
    assert gap_grounded > gap_plain


def test_build_mixture_pools_and_keeps_dataset_ids():
    """The mixture file must hold the union of the datasets, with ids and aligned columns."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        results = root / "results"
        results.mkdir()
        counts = {"alpha": 30, "beta": 20}
        for name, count in counts.items():
            for split in ("train", "val", "test"):
                data = make_split(count, hash((name, split)) % 1000, with_reference=split != "test")
                clean = {"dataset": name, "split": split, "indices": torch.arange(count), "labels": data["labels"],
                         "prompt_features": data["prompt_features"], "generated_features": data["generated_features"]}
                if split != "test":
                    clean["reference_features"] = data["reference_features"]
                torch.save(clean, root / f"{name}_{split}_2b.pt")
                torch.save({"indices": torch.arange(count), "prompt_features_corrupt": data["prompt_features_corrupt"],
                            "generated_features_corrupt": data["generated_features_corrupt"],
                            "confidence_clean": torch.rand(count), "confidence_corrupt": torch.rand(count)},
                           root / f"{name}_{split}_2b_corrupt.pt")
                lines = [
                    '{"index": %d, "prediction": {"confidence": 0.9, "answer": "ab"}}' % i for i in range(count)
                ]
                (results / f"{name}_{split}_2b_baseline.jsonl").write_text("\n".join(lines), encoding="utf-8")
        old = (build_mixture.FEATURE_DIR, build_mixture.RESULTS_DIR)
        build_mixture.FEATURE_DIR, build_mixture.RESULTS_DIR = root, results
        try:
            merged = build_mixture.build_split(["alpha", "beta"], "train", "2b", None, 0)
            test_split = build_mixture.build_split(["alpha", "beta"], "test", "2b", None, 0)
            capped = build_mixture.build_split(["alpha", "beta"], "train", "2b", 10, 0)
        finally:
            build_mixture.FEATURE_DIR, build_mixture.RESULTS_DIR = old
        assert len(merged["labels"]) == 50 and int((merged["dataset_id"] == 1).sum()) == 20
        assert "reference_features" in merged and "reference_features" not in test_split
        assert merged["answer_length"].min() == 2 and merged["dataset_names"] == ["alpha", "beta"]
        assert len(capped["labels"]) == 20


def test_quick_sweep_runs_for_every_head_kind():
    """The sweep returns winners, per-dataset test results, and grounding for each head kind."""
    train, validation, test = (tensors_of(make_split(n, s)) for n, s in ((400, 6), (150, 7), (150, 8)))
    for kind in ("pre", "verifier", "unified"):
        result, scores = sweep_mixture.run_sweep(
            kind, train, validation, test, ["a", "b"], DEVICE, quick=True, max_epochs=2, patience=2, batch_size=128
        )
        assert {"old", "new", "old_linear", "new_linear"} <= set(result["winners"])
        winner = result["winners"]["new"]
        assert winner["config"]["ground_weight"] > 0 and result["winners"]["old"]["config"]["ground_weight"] == 0
        assert set(winner["test_per_dataset"]) == {"a", "b"}
        mode = "joint" if kind == "unified" else "main"
        assert "mean" in winner["test"][mode]["correctness_auroc"] and "pooled" in winner["grounding"][mode]
        assert scores["new"][mode]["clean"].shape == (3, 150)
        if kind != "pre":
            assert {"cell_plain", "cell_contrast", "cell_ground", "cell_full", "new_full"} <= set(result["winners"])


def test_extra_corruption_grounding_is_reported():
    """A sweep given extra corruption features reports grounding against each of them, pooled and per dataset."""
    train, validation, test = (tensors_of(make_split(n, s)) for n, s in ((400, 6), (150, 7), (150, 8)))
    other = make_split(150, 9)
    extra = {"mask": {"prompt_corrupt": other["prompt_features_corrupt"], "generated_corrupt": other["generated_features_corrupt"]}}
    result, _ = sweep_mixture.run_sweep(
        "unified", train, validation, test, ["a", "b"], DEVICE, quick=True, max_epochs=2, patience=2, batch_size=128,
        extra_tests=extra,
    )
    entry = result["winners"]["new"]["grounding_extra"]["mask"]["joint"]
    assert "pooled" in entry and "a" in entry and "order_accuracy" in entry["pooled"]["correct"]


def test_build_extra_test_and_corruption_paths():
    """Pooled extra test files keep the row order of the clean split and name files by corruption."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name, count in {"alpha": 12, "beta": 8}.items():
            torch.save({"indices": torch.arange(count)}, root / f"{name}_test_2b.pt")
            torch.save({"indices": torch.arange(count), "prompt_features_corrupt": torch.randn(count, SIZE),
                        "generated_features_corrupt": torch.randn(count, SIZE), "confidence_corrupt": torch.rand(count)},
                       root / f"{name}_test_2b_corrupt_mask.pt")
        old = build_mixture.FEATURE_DIR
        build_mixture.FEATURE_DIR = root
        try:
            extra = build_mixture.build_extra_test(["alpha", "beta"], "2b", "mask")
            swap_name = build_mixture.corrupt_path("alpha", "test", "2b", "swap").name
            mask_name = build_mixture.corrupt_path("alpha", "test", "2b", "mask").name
        finally:
            build_mixture.FEATURE_DIR = old
        assert extra["prompt_features_corrupt"].shape == (20, SIZE) and len(extra["confidence_corrupt"]) == 20
        assert swap_name == "alpha_test_2b_corrupt.pt" and mask_name == "alpha_test_2b_corrupt_mask.pt"


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
