# test_config.py
#
# Smoke tests for the model and dataset config files. These do not touch
# the network or load any real model or dataset weights, they only check
# that the registries in model_config.py and each dataset_config.py are
# well-formed, so a typo in a repo id or a missing field is caught
# immediately instead of surfacing as a confusing error partway through
# a long download run.
#
# Run directly with `python -m tests.test_config`, or collect with
# pytest if it is installed.

from data.approach1_latent_self_correction import dataset_config as approach1_config
from data.approach2_verifier_reranking import dataset_config as approach2_config
from data.common.dataset_entry import DatasetEntry
from models.model_config import MODEL_VARIANTS, get_variant


def test_model_variants_have_repo_ids():
    """Every registered model size must have a well-formed Hugging Face repo id."""
    for size, variant in MODEL_VARIANTS.items():
        assert variant.hf_repo_id, f"model variant '{size}' is missing a repo id"
        assert "/" in variant.hf_repo_id, f"model variant '{size}' repo id looks malformed: {variant.hf_repo_id}"


def test_get_variant_rejects_unknown_size():
    """Looking up an unregistered model size should fail loudly, not silently."""
    try:
        get_variant("not-a-real-size")
    except ValueError:
        return
    raise AssertionError("get_variant should have raised ValueError for an unknown size")


def _check_dataset_group(group: dict, group_name: str) -> None:
    """Check that every entry in a dataset group is a well-formed DatasetEntry.

    A dataset is allowed to have hf_repo_id set to None, meaning it must
    be sourced manually, but if it is None, source_url must be set so
    there is still somewhere to point a teammate looking for the data.
    """
    for name, entry in group.items():
        assert isinstance(entry, DatasetEntry), f"{group_name}.{name} is not a DatasetEntry"
        if entry.hf_repo_id is None:
            assert entry.source_url, f"{group_name}.{name} has no Hub id and no source_url to fall back on"
        else:
            assert "/" in entry.hf_repo_id, f"{group_name}.{name} repo id looks malformed: {entry.hf_repo_id}"


def test_approach1_dataset_entries_are_well_formed():
    """Check every dataset group defined for Approach 1."""
    _check_dataset_group(approach1_config.TRAINING_DATASETS, "approach1.TRAINING_DATASETS")
    _check_dataset_group(approach1_config.TESTBED_DATASETS, "approach1.TESTBED_DATASETS")
    _check_dataset_group(approach1_config.OOD_DATASETS, "approach1.OOD_DATASETS")
    _check_dataset_group(approach1_config.QHEAD_DATASETS, "approach1.QHEAD_DATASETS")


def test_qhead_datasets_have_builders():
    """Every Q-head dataset in the config must have a split builder, and the other way around."""
    from data.approach1_latent_self_correction.build_qhead_splits import BUILDERS

    assert set(BUILDERS) == set(approach1_config.QHEAD_DATASETS)


def test_approach2_dataset_entries_are_well_formed():
    """Check every dataset group defined for Approach 2."""
    _check_dataset_group(approach2_config.TRAINING_DATASETS, "approach2.TRAINING_DATASETS")
    _check_dataset_group(approach2_config.OOD_DATASETS, "approach2.OOD_DATASETS")


def _run_all() -> None:
    """Run every test_* function in this module and report pass and fail counts.

    Lets this file run directly with `python -m tests.test_config` when
    pytest is not installed, while staying discoverable by pytest when
    it is.
    """
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
