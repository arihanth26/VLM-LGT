# build_mixture.py
#
# Pools the cached features of several datasets into one mixture, split by
# split: the mixture train is the union of the dataset train splits, and the
# same for validation and test. Each row keeps a dataset id so results can be
# reported pooled and per dataset. Clean states come from extract_q_features.py,
# corrupted-image states and the teacher-forced confidences from
# extract_corrupt_features.py, and the model's own generation confidence from the
# baseline output. Written to q_features/mixture_<split>_<size>.pt.

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


def load_one(dataset: str, split: str, size: str) -> dict:
    """Load and align the clean, corrupted, and baseline columns of one dataset split."""
    clean = torch.load(FEATURE_DIR / f"{dataset}_{split}_{size}.pt", map_location="cpu", weights_only=True)
    corrupt = torch.load(FEATURE_DIR / f"{dataset}_{split}_{size}_corrupt.pt", map_location="cpu", weights_only=True)
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


def build_split(datasets: list[str], split: str, size: str, cap: int | None, seed: int) -> dict:
    """Concatenate one split across datasets, optionally capping each dataset's rows."""
    parts = []
    for dataset in datasets:
        part = load_one(dataset, split, size)
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


def main() -> None:
    """Build and save the mixture files for every split."""
    parser = argparse.ArgumentParser(description="Pool cached features into a dataset mixture.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--size", default="2b")
    parser.add_argument("--cap-train", type=int, default=None, help="Keep at most N training rows per dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="mixture")
    args = parser.parse_args()

    for split in SPLITS:
        merged = build_split(list(args.datasets), split, args.size, args.cap_train, args.seed)
        output = FEATURE_DIR / f"{args.tag}_{split}_{args.size}.pt"
        torch.save(merged, output)
        counts = ", ".join(
            f"{name} {int((merged['dataset_id'] == i).sum())}" for i, name in enumerate(args.datasets)
        )
        print(f"[done] {output.name}: {len(merged['labels'])} rows ({counts}), accuracy {float(merged['labels'].mean()):.3f}")


if __name__ == "__main__":
    main()
