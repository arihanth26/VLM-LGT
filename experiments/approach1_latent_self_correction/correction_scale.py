# correction_scale.py
#
# Helpers for running the correction experiments (prompted retry, best-of-five selection,
# deeper pass) on the whole test splits instead of 100 examples. They load the test split
# that the Q-head pipeline already cached on disk, so a job needs no network access and
# scores exactly the questions the heads were evaluated on, and they cut a split into
# contiguous shards so one split can run as several GPU jobs in parallel. The shard outputs
# are merged afterwards by merge_correction_shards.py, which computes the policies that need
# the whole set (the lowest 20% by score, the random control) once, over all examples.

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "approach1_latent_self_correction" / "raw"
DATASET_NAMES = ("chartqa_full", "textvqa", "docvqa", "scienceqa_img")
# Datasets the heads and rankers never trained on, used only to test transfer (build_unseen_splits.py).
# They live in a split of their own, "unseen", so no file of theirs can mix into the in-domain train, val
# or test sets, and they come after DATASET_NAMES so every existing seed and index is unchanged.
UNSEEN_DATASETS = ("ai2d", "infovqa")
ALL_DATASETS = DATASET_NAMES + UNSEEN_DATASETS
SPLITS = ("train", "val", "test", "unseen")

# The frozen model the whole pipeline runs on. The default, 2B, is the main study. VLMLGT_MODEL=4b reruns the
# pipeline with Qwen3-VL-4B as the main model: every result folder moves under results/base_4b/ so no 2B file
# is ever read or overwritten by mistake, and the correctness heads (trained on 2B states, which have another
# width) are switched off, so their scores are a constant 0.5 and the 4B ranker is trained without them.
MODEL_SIZE = os.environ.get("VLMLGT_MODEL", "2b").lower()
EXPERIMENT_DIR = Path(__file__).parent
BASE_RESULTS = EXPERIMENT_DIR / "results"  # the main (2B) results; also defines the ranker's train subset
RESULTS_ROOT = BASE_RESULTS if MODEL_SIZE == "2b" else BASE_RESULTS / f"base_{MODEL_SIZE}"
HEADS_AVAILABLE = MODEL_SIZE == "2b"
# VLMLGT_CHAIN=1: every stage (pools, first round, second round, sampled reasoning) takes shard k as the k-th of
# every num_shards questions of ranker_subset, so one job can run all stages for its shard in sequence: each stage
# finds exactly the questions its own earlier stage produced. Off by default (the 2B runs used per-stage shards).
CHAIN = os.environ.get("VLMLGT_CHAIN") == "1"


def ranker_subset(dataset: str, split: str) -> list[int]:
    """Indices of the questions the 2B second round covered (for train: the ranker's 12,000), so a rerun with
    another model answers exactly the same questions."""
    import json

    indices = set()
    for path in sorted((BASE_RESULTS / "correction_v2" / split).glob(f"{dataset}_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and '"judge"' in line:
                indices.add(json.loads(line)["index"])
    return sorted(indices)


def chain_indices(dataset: str, split: str, shard: int, num_shards: int, max_examples: int | None = None) -> list[int]:
    """The questions of one shard under VLMLGT_CHAIN: every num_shards-th question of ranker_subset."""
    return ranker_subset(dataset, split)[shard::num_shards][:max_examples]


def pool_temperatures(pool_dir: Path, split: str) -> tuple[str, ...]:
    """Temperature folders that hold pools for this split (2B val/test: 1 and 1.3; train, unseen, 4B: 1 only)."""
    return tuple(t for t in ("1", "1.3") if (pool_dir / f"T{t}" / split).is_dir())


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
