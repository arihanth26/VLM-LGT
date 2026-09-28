# utils.py
#
# Small shared helpers used by both approach 1 and approach 2 dataset
# scripts, so download and preprocessing code does not repeat the same
# file handling logic in two places.

import json
import os


def ensure_dir(path: str) -> str:
    """Create a directory if it does not already exist and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def write_jsonl(records: list, path: str) -> None:
    """Write a list of dictionaries to a JSON Lines file.

    Each dataset's preprocess script converts its own raw format into
    plain dictionaries with a shared set of keys (image, question,
    answer, and so on). This function writes those records to disk in
    the format the rest of the project reads them back in.
    """
    parent_dir = os.path.dirname(path) or "."
    ensure_dir(parent_dir)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_jsonl(path: str) -> list:
    """Read a JSON Lines file back into a list of dictionaries."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
