# download.py
#
# Downloads the raw datasets listed in dataset_config.py for Approach 2
# and caches them locally under data/approach2_verifier_reranking/raw.
# Both the training and out-of-distribution groups are pulled directly
# from the Hugging Face Hub; the verifier's failure-inclusive pool is not
# downloaded here, see experiments/approach2_verifier_reranking/build_failure_pool.py
# for that. Each download is
# wrapped so one broken or renamed dataset does not stop the rest of the
# run, since repo ids and schemas on the Hub can change after this file
# was last verified (see DATASETS.md for that date).

import argparse
import os

from datasets import load_dataset

from data.approach2_verifier_reranking.dataset_config import (
    OOD_DATASETS,
    TRAINING_DATASETS,
)
from data.common.dataset_entry import DatasetEntry
from data.common.utils import ensure_dir

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download_dataset(name: str, entry: DatasetEntry, out_dir: str) -> None:
    """Download one dataset from the Hugging Face Hub and save it to disk.

    Skips the download if the target directory already exists and is
    not empty, so re-running this script does not re-download data that
    is already available locally. Passes entry.hf_config through when a
    dataset requires an explicit config name (Visual Genome, for
    example). Failures are caught and reported with the dataset's source
    URL instead of crashing the whole run.
    """
    target_dir = os.path.join(out_dir, name)
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[skip] {name}: already downloaded at {target_dir}")
        return

    config_suffix = f" (config: {entry.hf_config})" if entry.hf_config else ""
    print(f"[download] {name} from {entry.hf_repo_id}{config_suffix}")

    try:
        if entry.hf_config:
            dataset = load_dataset(entry.hf_repo_id, name=entry.hf_config)
        else:
            dataset = load_dataset(entry.hf_repo_id)
    except Exception as exc:
        print(
            f"[error] {name}: failed to download ({exc}). "
            f"Check {entry.source_url} for the current dataset card, "
            f"the repo id, config name, or schema may have changed."
        )
        return

    ensure_dir(target_dir)
    dataset.save_to_disk(target_dir)


def download_all(groups: list) -> None:
    """Download every dataset in the requested groups from dataset_config.py.

    groups is a list of names among "training", "ood", so a caller can
    download only what they currently need instead of everything at
    once.
    """
    ensure_dir(RAW_DIR)

    if "training" in groups:
        for name, entry in TRAINING_DATASETS.items():
            download_dataset(name, entry, RAW_DIR)

    if "ood" in groups:
        for name, entry in OOD_DATASETS.items():
            download_dataset(name, entry, RAW_DIR)


def main() -> None:
    """Parse command line arguments and trigger the requested downloads."""
    parser = argparse.ArgumentParser(description="Download Approach 2 datasets.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["training", "ood"],
        choices=["training", "ood"],
        help="Which dataset groups to download.",
    )
    args = parser.parse_args()
    download_all(args.groups)


if __name__ == "__main__":
    main()
