# build_mixture.py
#
# Pools the cached features of several datasets into one mixture, split by
# split: the mixture train is the union of the dataset train splits, and the
# same for validation and test. Each row keeps a dataset id so results can be
# reported pooled and per dataset. Clean states come from extract_q_features.py,
# corrupted-image states and the teacher-forced confidences from
# extract_corrupt_features.py, and the model's own generation confidence from the
# baseline output. Written to q_features/<tag>_<split>_<size>.pt.
#
# The corrupted-image states of any corruption can be pooled (--corruption), a single
# dataset can be built as its own one-dataset mixture (--datasets), and test-only files
# of other corruptions can be pooled in the same row order (--extra-test), which is how
# grounding learned under one corruption is checked against another.

import argparse
import json
from pathlib import Path

import torch


FEATURE_DIR = Path(__file__).parent / "q_features"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_DATASETS = ("chartqa_full", "textvqa", "docvqa", "scienceqa_img")
SPLITS = ("train", "val", "test")


def baseline_columns(path: Path, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Read generation confidence and answer length for the given dataset indices."""
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "prediction" in record:
                records[record["index"]] = record["prediction"]
    confidence = torch.tensor([records[i]["confidence"] for i in indices], dtype=torch.float32)
    length = torch.tensor([len(str(records[i]["answer"])) for i in indices], dtype=torch.float32)
    return confidence, length


def corrupt_path(dataset: str, split: str, size: str, corruption: str) -> Path:
    """Where the corrupted-image features of one dataset split and corruption are cached."""
    suffix = "" if corruption == "swap" else f"_{corruption}"
    return FEATURE_DIR / f"{dataset}_{split}_{size}_corrupt{suffix}.pt"


def load_one(dataset: str, split: str, size: str, corruption: str = "swap") -> dict:
    """Load and align the clean, corrupted, and baseline columns of one dataset split."""
    clean = torch.load(FEATURE_DIR / f"{dataset}_{split}_{size}.pt", map_location="cpu", weights_only=True)
    corrupt = torch.load(corrupt_path(dataset, split, size, corruption), map_location="cpu", weights_only=True)
    if not torch.equal(clean["indices"], corrupt["indices"]):
        raise ValueError(f"{dataset} {split}: clean and corrupted features cover different examples")
    indices = clean["indices"].tolist()
    confidence, length = baseline_columns(RESULTS_DIR / f"{dataset}_{split}_{size}_baseline.jsonl", indices)
    part = {
        "indices": clean["indices"],
        "labels": clean["labels"],
        "prompt_features": clean["prompt_features"],
        "generated_features": clean["generated_features"],
        "prompt_features_corrupt": corrupt["prompt_features_corrupt"],
        "generated_features_corrupt": corrupt["generated_features_corrupt"],
        "token_confidence": confidence,
        "confidence_clean": corrupt["confidence_clean"],
        "confidence_corrupt": corrupt["confidence_corrupt"],
        "answer_length": length,
    }
    if "reference_features" in clean:
        part["reference_features"] = clean["reference_features"]
    return part


def build_split(datasets: list[str], split: str, size: str, cap: int | None, seed: int, corruption: str = "swap") -> dict:
    """Concatenate one split across datasets, optionally capping each dataset's rows."""
    parts = []
    for dataset in datasets:
        part = load_one(dataset, split, size, corruption)
        if cap is not None and split == "train" and len(part["labels"]) > cap:
            keep = torch.randperm(len(part["labels"]), generator=torch.Generator().manual_seed(seed))[:cap].sort().values
            part = {key: value[keep] for key, value in part.items()}
        parts.append(part)
    keys = set.intersection(*(set(part) for part in parts))
    merged = {key: torch.cat([part[key] for part in parts]) for key in sorted(keys)}
    merged["dataset_id"] = torch.cat(
        [torch.full((len(part["labels"]),), position, dtype=torch.long) for position, part in enumerate(parts)]
    )
    merged["dataset_names"] = list(datasets)
    merged["split"] = split
    return merged


def build_extra_test(datasets: list[str], size: str, corruption: str) -> dict:
    """Pool the corrupted test features of another corruption, in the same row order as the mixture test split."""
    parts = []
    for dataset in datasets:
        clean = torch.load(FEATURE_DIR / f"{dataset}_test_{size}.pt", map_location="cpu", weights_only=True)
        corrupt = torch.load(corrupt_path(dataset, "test", size, corruption), map_location="cpu", weights_only=True)
        if not torch.equal(clean["indices"], corrupt["indices"]):
            raise ValueError(f"{dataset} test: clean and {corruption} features cover different examples")
        parts.append(corrupt)
    return {
        "corruption": corruption,
        "prompt_features_corrupt": torch.cat([p["prompt_features_corrupt"] for p in parts]),
        "generated_features_corrupt": torch.cat([p["generated_features_corrupt"] for p in parts]),
        "confidence_corrupt": torch.cat([p["confidence_corrupt"] for p in parts]),
    }


def main() -> None:
    """Build and save the mixture files for every split."""
    parser = argparse.ArgumentParser(description="Pool cached features into a dataset mixture.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--size", default="2b")
    parser.add_argument("--cap-train", type=int, default=None, help="Keep at most N training rows per dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="mixture")
    parser.add_argument("--corruption", default="swap", help="Which corrupted-image features to pool.")
    parser.add_argument("--extra-test", nargs="*", default=[], help="Also pool the test features of these other corruptions.")
    parser.add_argument("--extra-only", action="store_true", help="Only write the extra test files.")
    args = parser.parse_args()

    for mode in args.extra_test:
        extra = build_extra_test(list(args.datasets), args.size, mode)
        output = FEATURE_DIR / f"{args.tag}_test_{args.size}_corrupt_{mode}.pt"
        torch.save(extra, output)
        print(f"[done] {output.name}: {len(extra['confidence_corrupt'])} rows")
    if args.extra_only:
        return
    for split in SPLITS:
        merged = build_split(list(args.datasets), split, args.size, args.cap_train, args.seed, args.corruption)
        output = FEATURE_DIR / f"{args.tag}_{split}_{args.size}.pt"
        torch.save(merged, output)
        counts = ", ".join(
            f"{name} {int((merged['dataset_id'] == i).sum())}" for i, name in enumerate(args.datasets)
        )
        print(f"[done] {output.name}: {len(merged['labels'])} rows ({counts}), accuracy {float(merged['labels'].mean()):.3f}")


if __name__ == "__main__":
    main()
