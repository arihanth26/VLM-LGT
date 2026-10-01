# download.py
#
# Downloads the raw datasets listed in dataset_config.py for Approach 1
# and caches them locally under data/approach1_latent_self_correction/raw.
# Datasets available on the Hugging Face Hub are pulled automatically;
# Ref-Adv-S has no Hub id and is checked for as a manual drop-in instead.
# Each download is wrapped so one broken or renamed dataset does not stop
# the rest of the run, since repo ids and schemas on the Hub can change
# after this file was last verified (see DATASETS.md for that date).

import argparse
import os

from datasets import load_dataset

from data.approach1_latent_self_correction.dataset_config import (
    OOD_DATASETS,
    TESTBED_DATASETS,
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


def check_manual_dataset(name: str, entry: DatasetEntry, out_dir: str) -> None:
    """Check whether a manually sourced dataset has already been placed on disk.

    Ref-Adv-S is not distributed through the Hugging Face Hub, so this
    project expects it to be placed manually under raw/<name> before the
    testbed can be built. This function only reports whether that has
    happened; it does not fetch anything.
    """
    target_dir = os.path.join(out_dir, name)
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[ok] {name}: found local copy at {target_dir}")
    else:
        print(
            f"[missing] {name}: no local copy found at {target_dir}. "
            f"This dataset requires a manual download from {entry.source_url}, "
            f"place it there before running preprocess.py for the testbed split."
        )


def download_all(groups: list, dataset_names: list[str] | None = None) -> None:
    """Download every dataset in the requested groups from dataset_config.py.

    groups is a list of names among "training", "testbed", "ood", so a
    caller can download only what they currently need instead of
    everything at once.
    """
    ensure_dir(RAW_DIR)

    requested = set(dataset_names) if dataset_names else None

    if requested:
        available = set(TRAINING_DATASETS) | set(TESTBED_DATASETS) | set(OOD_DATASETS)
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown dataset names: {', '.join(sorted(unknown))}")

    if "training" in groups:
        for name, entry in TRAINING_DATASETS.items():
            if requested is None or name in requested:
                download_dataset(name, entry, RAW_DIR)

    if "testbed" in groups:
        for name, entry in TESTBED_DATASETS.items():
            if requested is not None and name not in requested:
                continue
            if entry.hf_repo_id is None:
                check_manual_dataset(name, entry, RAW_DIR)
            else:
                download_dataset(name, entry, RAW_DIR)

    if "ood" in groups:
        for name, entry in OOD_DATASETS.items():
            if requested is None or name in requested:
                download_dataset(name, entry, RAW_DIR)


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
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset keys to download instead of every dataset in the groups.",
    )
    args = parser.parse_args()
    download_all(args.groups, args.datasets)


if __name__ == "__main__":
    main()
