# download.py
#
# Downloads the raw datasets listed in dataset_config.py for Approach 1
# and caches them locally under data/approach1_latent_self_correction/raw.
# Datasets available on the Hugging Face Hub are pulled automatically;
# datasets that require a manual request (Ref-Adv-S, Ref-L4) are checked
# for and reported as missing rather than silently skipped.

import argparse
import os

from datasets import load_dataset

from data.approach1_latent_self_correction.dataset_config import (
    OOD_DATASETS,
    TESTBED_DATASETS,
    TRAINING_DATASETS,
)
from data.common.utils import ensure_dir

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download_dataset(name: str, hf_repo_id: str, out_dir: str) -> None:
    """Download one dataset from the Hugging Face Hub and save it to disk.

    Skips the download if the target directory already exists and is
    not empty, so re-running this script does not re-download data that
    is already available locally.
    """
    target_dir = os.path.join(out_dir, name)
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[skip] {name}: already downloaded at {target_dir}")
        return

    print(f"[download] {name} from {hf_repo_id}")
    dataset = load_dataset(hf_repo_id)
    ensure_dir(target_dir)
    dataset.save_to_disk(target_dir)


def check_manual_dataset(name: str, out_dir: str) -> None:
    """Check whether a manually sourced dataset has already been placed on disk.

    Ref-Adv-S and Ref-L4 are not distributed through the Hugging Face
    Hub, so this project expects them to be placed manually under
    raw/<name> before the testbed can be built. This function only
    reports whether that has happened; it does not fetch anything.
    """
    target_dir = os.path.join(out_dir, name)
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[ok] {name}: found local copy at {target_dir}")
    else:
        print(
            f"[missing] {name}: no local copy found at {target_dir}. "
            f"This dataset requires a manual download, place it there before "
            f"running preprocess.py for the testbed split."
        )


def download_all(groups: list) -> None:
    """Download every dataset in the requested groups from dataset_config.py.

    groups is a list of names among "training", "testbed", "ood", so a
    caller can download only what they currently need instead of
    everything at once.
    """
    ensure_dir(RAW_DIR)

    if "training" in groups:
        for name, hf_repo_id in TRAINING_DATASETS.items():
            download_dataset(name, hf_repo_id, RAW_DIR)

    if "testbed" in groups:
        for name, hf_repo_id in TESTBED_DATASETS.items():
            if hf_repo_id is None:
                check_manual_dataset(name, RAW_DIR)
            else:
                download_dataset(name, hf_repo_id, RAW_DIR)

    if "ood" in groups:
        for name, hf_repo_id in OOD_DATASETS.items():
            download_dataset(name, hf_repo_id, RAW_DIR)


def main() -> None:
    """Parse command line arguments and trigger the requested downloads."""
    parser = argparse.ArgumentParser(description="Download Approach 1 datasets.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["training", "testbed", "ood"],
        choices=["training", "testbed", "ood"],
        help="Which dataset groups to download.",
    )
    args = parser.parse_args()
    download_all(args.groups)


if __name__ == "__main__":
    main()
