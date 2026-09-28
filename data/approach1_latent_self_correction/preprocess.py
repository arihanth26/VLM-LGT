# preprocess.py
#
# Converts the raw datasets downloaded by download.py into a shared JSON
# Lines format that the rest of Approach 1 (Latent Self-Correction) can
# train and evaluate on. Every output record has the same core keys
# (image, question, answer, bbox) regardless of which source dataset it
# came from, so downstream training code does not need to special case
# each dataset.
#
# Field names in the raw Hugging Face datasets can change between
# versions. If a dataset's column names differ from what is assumed
# here, update the relevant normalize_* function rather than the
# training code.

import argparse
import os

from datasets import load_from_disk

from data.approach1_latent_self_correction.dataset_config import (
    OOD_DATASETS,
    TRAINING_DATASETS,
)
from data.common.utils import write_jsonl

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def normalize_grounding_example(example: dict, source: str) -> dict:
    """Normalize one grounding example (GQA, Visual Genome, RefCOCO style).

    Grounding datasets provide a bounding box alongside the question and
    answer, which Approach 1 needs to score whether a latent correction
    moved toward or away from the correct region.
    """
    return {
        "source": source,
        "image": example.get("image"),
        "question": example.get("question") or example.get("sentence"),
        "answer": example.get("answer"),
        "bbox": example.get("bbox") or example.get("box"),
    }


def normalize_vqa_example(example: dict, source: str) -> dict:
    """Normalize one open-ended or chart and diagram QA example.

    Used for the harder, out-of-distribution evaluation sets (ChartQA,
    AI2D, MathVista, V*Bench, HR-Bench, VSR), which do not include a
    bounding box.
    """
    return {
        "source": source,
        "image": example.get("image"),
        "question": example.get("question"),
        "answer": example.get("answer"),
        "bbox": None,
    }


def preprocess_group(dataset_names: dict, normalize_fn, split_out_dir: str) -> None:
    """Load each raw dataset in a group and write it out in the shared format.

    dataset_names maps a short dataset name to its Hugging Face repo id
    (only the keys are used here, the raw data itself is read from
    disk). normalize_fn is the function used to convert one raw example
    into the shared record schema for this group of datasets.
    """
    for name in dataset_names:
        raw_path = os.path.join(RAW_DIR, name)
        if not os.path.isdir(raw_path):
            print(f"[skip] {name}: not downloaded yet, run download.py first")
            continue

        print(f"[preprocess] {name}")
        dataset = load_from_disk(raw_path)

        for split_name, split_data in dataset.items():
            records = [normalize_fn(example, name) for example in split_data]
            out_path = os.path.join(split_out_dir, f"{name}_{split_name}.jsonl")
            write_jsonl(records, out_path)
            print(f"  wrote {len(records)} examples to {out_path}")


def main() -> None:
    """Parse command line arguments and run preprocessing for the chosen groups."""
    parser = argparse.ArgumentParser(description="Preprocess Approach 1 datasets.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["training", "ood"],
        choices=["training", "ood"],
        help=(
            "Which dataset groups to preprocess. The testbed group uses "
            "manually sourced data (Ref-Adv-S, Ref-L4) and should be "
            "prepared separately once those files are placed under raw/."
        ),
    )
    args = parser.parse_args()

    if "training" in args.groups:
        preprocess_group(TRAINING_DATASETS, normalize_grounding_example, PROCESSED_DIR)
    if "ood" in args.groups:
        preprocess_group(OOD_DATASETS, normalize_vqa_example, PROCESSED_DIR)


if __name__ == "__main__":
    main()
