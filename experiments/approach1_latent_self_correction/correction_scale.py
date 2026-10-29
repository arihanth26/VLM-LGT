# correction_scale.py
#
# Helpers for running the correction experiments (prompted retry, best-of-five selection,
# deeper pass) on the whole test splits instead of 100 examples. They load the test split
# that the Q-head pipeline already cached on disk, so a job needs no network access and
# scores exactly the questions the heads were evaluated on, and they cut a split into
# contiguous shards so one split can run as several GPU jobs in parallel. The shard outputs
# are merged afterwards by merge_correction_shards.py, which computes the policies that need
# the whole set (the lowest 20% by score, the random control) once, over all examples.

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "approach1_latent_self_correction" / "raw"
DATASET_NAMES = ("chartqa_full", "textvqa", "docvqa", "scienceqa_img")


def dataset_dir(name: str) -> Path:
    """The folder that holds one dataset's cached splits and image files."""
    return RAW_DIR / name


def cached_test_split(name: str):
    """The full test split of one dataset, loaded from the cache written by the Q-head pipeline."""
    from datasets import DatasetDict, load_from_disk

    loaded = load_from_disk(str(dataset_dir(name)))
    if not isinstance(loaded, DatasetDict) or "test" not in loaded:
        raise ValueError(f"{name} does not contain a test split")
    return loaded["test"]


def shard_range(count: int, shard: int, num_shards: int, max_examples: int | None = None) -> range:
    """The contiguous block of example indices that belongs to one shard, optionally capped for a smoke test."""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} is outside 0..{num_shards - 1}")
    start, stop = count * shard // num_shards, count * (shard + 1) // num_shards
    if max_examples is not None:
        stop = min(stop, start + max_examples)
    return range(start, stop)
