# repair_v2_shards.py
#
# Finds second-round records whose new answers have no stored state (a job killed after writing the
# JSONL line but before saving the .pt), removes those records so a resubmitted shard regenerates
# them, and prints which shards need a rerun. Run on a GPU node like every other job.

import json
from pathlib import Path

import torch

V2_DIR = Path(__file__).parent / "results" / "correction_v2"


def main() -> None:
    """Drop incomplete records from every shard and list the shards to resubmit."""
    for path in sorted(V2_DIR.glob("*/*.jsonl")):
        saved = torch.load(path.with_suffix(".pt"), weights_only=False)["questions"] if path.with_suffix(".pt").exists() else {}
        kept, dropped = [], 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            stored = set(saved.get(record["index"], {}).get("answers", []))
            if "encoded" in record and not set(record["encoded"]) <= stored:
                dropped += 1
                continue
            kept.append(line)
        if dropped:
            path.write_text("\n".join(kept) + "\n")
            print(f"[repair] {path.parent.name}/{path.name}: removed {dropped} incomplete records", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
