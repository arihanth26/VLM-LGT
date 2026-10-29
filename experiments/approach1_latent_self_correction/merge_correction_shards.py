# merge_correction_shards.py
#
# Merges the shard outputs of the full-split correction experiments into one result per
# experiment, in the same files and formats the 100-example runs produced (records JSONL,
# summary JSON, and for best-of-five an HTML report). The policies that depend on the whole
# set, such as retrying the lowest 20% by head score or the matched random control, are
# computed here once over all examples, never per shard. Each experiment reads the shard files
# under results/full/<experiment>/ and writes results/mixture_test_2b_<name>_full.*.
#   best5   best-of-five candidate selection        -> mixture_test_2b_best_of_five_full
#   retry   prompted retry with scored reranking    -> mixture_test_2b_prompted_retry_reranked_full
#   deep    deeper pass after best-of-five          -> mixture_test_2b_best_of_five_deep_full
# Usage: python -m experiments.approach1_latent_self_correction.merge_correction_shards [--experiments best5 retry deep]

import argparse
import json
from pathlib import Path

from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES

RESULTS = Path(__file__).parent / "results"
FULL = RESULTS / "full"
NAMES = {
    "best5": "mixture_test_2b_best_of_five_full",
    "retry": "mixture_test_2b_prompted_retry_reranked_full",
    "deep": "mixture_test_2b_best_of_five_deep_full",
}
DEFAULT_HEAD = Path(__file__).resolve().parents[2] / "trained_heads" / "approach1" / "winners" / "mixture" / "unified_new_seed1.pt"


def read_shards(folder: Path) -> list[dict]:
    """Every record of every shard file in a folder, one per example, ordered by dataset then index."""
    records = {}
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[(record["dataset"], record["index"])] = record
    order = {name: position for position, name in enumerate(DATASET_NAMES)}
    return [records[key] for key in sorted(records, key=lambda key: (order.get(key[0], 99), key[1]))]


def write_records(path: Path, records: list[dict]) -> None:
    """Write merged records as JSON lines."""
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, default=str) + "\n")


def count_report(name: str, records: list[dict], success_key: str) -> None:
    """Print how many examples each dataset contributed and how many failed."""
    ok = [r for r in records if success_key in r]
    print(f"[{name}] {len(ok)} examples merged, {len(records) - len(ok)} errors; per dataset: "
          + ", ".join(f"{d}={sum(1 for r in ok if r['dataset'] == d)}" for d in DATASET_NAMES))


def merge_best5(args) -> None:
    """Merge best-of-five shards, compute the selection summary, and write the HTML report."""
    from experiments.approach1_latent_self_correction.candidate_reranking import build_summary, write_html

    records = read_shards(FULL / "best5")
    count_report("best5", records, "candidates")
    successful = [r for r in records if "candidates" in r]
    output = RESULTS / f"{NAMES['best5']}.jsonl"
    write_records(output, successful + [r for r in records if "candidates" not in r])
    metadata = {"head_path": str(args.head), "candidate_count": args.candidates, "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed, "limit": None, "full_test_split": True}
    summary = build_summary(successful, metadata)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_html(successful, summary, output.with_suffix(".html"))
    print(f"[done] wrote {output}")


def merge_retry(args) -> None:
    """Merge retry shards and compute every retry policy over the full set."""
    from experiments.approach1_latent_self_correction.online_prompted_retry import add_policy_selections, write_outputs

    records = read_shards(FULL / "retry")
    count_report("retry", records, "baseline_correct")
    successful = [r for r in records if "baseline_correct" in r]
    add_policy_selections(successful, args.threshold, args.review_fraction, args.random_seed)
    write_outputs(RESULTS / f"{NAMES['retry']}.jsonl", successful + [r for r in records if "baseline_correct" not in r],
                  {"head_path": str(args.head), "threshold": args.threshold, "review_fraction": args.review_fraction, "random_seed": args.random_seed, "limit": None, "full_test_split": True})


def merge_deep(args) -> None:
    """Merge deeper-pass shards and compute the replacement policies."""
    from experiments.approach1_latent_self_correction.deepen_selected_candidate import add_selections, write_summary

    records = read_shards(FULL / "deep")
    count_report("deep", records, "deep_answer")
    successful = [r for r in records if "deep_answer" in r]
    add_selections(successful)
    output = RESULTS / f"{NAMES['deep']}.jsonl"
    write_records(output, successful + [r for r in records if "deep_answer" not in r])
    write_summary(successful, RESULTS / f"{NAMES['best5']}.jsonl", output)


def main() -> None:
    """Parse arguments and merge the requested experiments."""
    parser = argparse.ArgumentParser(description="Merge the shard outputs of the full-split correction experiments.")
    parser.add_argument("--experiments", nargs="+", choices=("best5", "retry", "deep"), default=["best5", "retry", "deep"])
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--review-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    for name in args.experiments:
        {"best5": merge_best5, "retry": merge_retry, "deep": merge_deep}[name](args)


if __name__ == "__main__":
    main()
