# prompted_retry.py
#
# Tests a simple correction policy on the pooled Approach 1 test set. The
# exported unified head scores cached baseline features, and low-confidence
# answers are regenerated with an explicit request to inspect the image again.

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk

from experiments.approach1_latent_self_correction.baseline import (
    extract_image,
    extract_question,
    extract_reference,
    score_prediction,
)
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from models.load_model import load_qwen3_vl
from models.run_inference import generate_with_confidence

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "approach1_latent_self_correction" / "raw"
FEATURE_DIR = Path(__file__).parent / "q_features"
BASELINE_DIR = Path(__file__).parent / "results"
DEFAULT_HEAD = REPO_ROOT / "trained_heads" / "approach1" / "winners" / "mixture" / "unified_new_seed1.pt"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "mixture_test_2b_prompted_retry.jsonl"
RETRY_INSTRUCTION = (
    "Your previous answer was: {answer}\n"
    "Try the question again. Carefully inspect the visual evidence, including labels, "
    "values, and spatial relationships. Do not assume the previous answer is correct.\n"
    "Respond with only the final answer, without explanation."
)


def load_jsonl_by_index(path: Path) -> dict[int, dict]:
    """Index successful JSONL records by their source dataset index."""
    with path.open("r", encoding="utf-8") as source:
        return {
            record["index"]: record
            for line in source
            if line.strip()
            for record in [json.loads(line)]
            if "prediction" in record
        }


def confidence_mask(confidences: np.ndarray, threshold: float, review_fraction: float | None) -> np.ndarray:
    """Select scores below a cutoff, or the requested lowest-scoring fraction."""
    values = np.asarray(confidences, dtype=float)
    if review_fraction is None:
        return values < threshold
    if not 0.0 <= review_fraction <= 1.0:
        raise ValueError("review_fraction must be between 0 and 1")
    count = int(np.ceil(len(values) * review_fraction))
    selected = np.zeros(len(values), dtype=bool)
    if count:
        selected[np.argsort(values, kind="stable")[:count]] = True
    return selected


def retry_question(question: str, previous_answer: str) -> str:
    """Build the second-pass prompt while retaining the original user question."""
    return f"{question}\n{RETRY_INSTRUCTION.format(answer=previous_answer)}"


def summarize(records: list[dict], selection_key: str = "flagged") -> dict:
    """Measure coverage and accuracy for one retry-selection policy."""
    successful = [record for record in records if "baseline_correct" in record]
    flagged = [record for record in successful if record[selection_key]]
    retried = [record for record in flagged if "retry_correct" in record]
    baseline_correct = sum(record["baseline_correct"] for record in successful)
    final_correct = sum(
        record["retry_correct"]
        if record[selection_key] and "retry_correct" in record
        else record["baseline_correct"]
        for record in successful
    )
    fixed = sum(not record["baseline_correct"] and record["retry_correct"] for record in retried)
    broken = sum(record["baseline_correct"] and not record["retry_correct"] for record in retried)
    return {
        "examples": len(successful),
        "flagged": len(flagged),
        "retried": len(retried),
        "coverage": len(flagged) / len(successful) if successful else 0.0,
        "baseline_accuracy": baseline_correct / len(successful) if successful else 0.0,
        "final_accuracy": final_correct / len(successful) if successful else 0.0,
        "fixed_errors": fixed,
        "broken_answers": broken,
        "net_corrections": fixed - broken,
    }


def policy_summary(records: list[dict], selection_key: str) -> dict:
    """Report one policy both pooled and separately for each source dataset."""
    dataset_names = sorted({record["dataset"] for record in records if "dataset" in record})
    return {
        "pooled": summarize(records, selection_key),
        "per_dataset": {
            name: summarize([record for record in records if record.get("dataset") == name], selection_key)
            for name in dataset_names
        },
    }


def write_report(output_path: Path, metadata: dict, policies: dict[str, str]) -> None:
    """Write aggregate metrics for every retry policy beside the JSONL output."""
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {
        **metadata,
        "policies": {name: policy_summary(records, key) for name, key in policies.items()},
    }
    report_path = output_path.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {json.dumps(report, sort_keys=True)}")
    print(f"[done] wrote {output_path} and {report_path}")


def run_prompted_retry(
    feature_path: Path,
    head_path: Path,
    baseline_dir: Path,
    raw_dir: Path,
    output_path: Path,
    size: str,
    threshold: float,
    review_fraction: float | None,
    retry_all: bool,
    random_seed: int,
    max_new_tokens: int,
    limit: int | None,
) -> None:
    """Score the mixture baseline and retry only examples selected by the head."""
    mixture = torch.load(feature_path, map_location="cpu", weights_only=True)
    head, spec, temperatures = load_head(head_path)
    probabilities = score(
        head,
        spec,
        temperatures,
        mixture["prompt_features"],
        mixture["generated_features"],
    )["joint"]
    row_count = len(probabilities) if limit is None else min(limit, len(probabilities))
    rows = list(range(row_count))
    active_probabilities = probabilities[:row_count]
    active_token_confidence = mixture["token_confidence"][:row_count].float().cpu().numpy()
    threshold_selected = confidence_mask(active_probabilities, threshold, None)
    fraction = 0.2 if review_fraction is None else review_fraction
    fraction_selected = confidence_mask(active_probabilities, threshold, fraction)
    token_selected = confidence_mask(active_token_confidence, threshold, fraction)
    generator = np.random.default_rng(random_seed)
    random_selected = np.zeros(row_count, dtype=bool)
    random_count = int(fraction_selected.sum())
    if random_count:
        random_selected[generator.choice(row_count, random_count, replace=False)] = True
    generate_retry = np.ones(row_count, dtype=bool) if retry_all else (
        fraction_selected if review_fraction is not None else threshold_selected
    )

    dataset_names = list(mixture["dataset_names"])
    baselines = {
        name: load_jsonl_by_index(baseline_dir / f"{name}_test_{size}_baseline.jsonl")
        for name in dataset_names
    }
    datasets = {}
    for name in dataset_names:
        loaded = load_from_disk(str(raw_dir / name))
        if not isinstance(loaded, DatasetDict) or "test" not in loaded:
            raise ValueError(f"{name} does not contain a test split")
        datasets[name] = loaded["test"]

    completed = set()
    if output_path.exists():
        completed = {
            (record["dataset"], record["index"])
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for record in [json.loads(line)]
        }
    pending = [
        row
        for row in rows
        if (dataset_names[int(mixture["dataset_id"][row])], int(mixture["indices"][row])) not in completed
    ]
    needs_model = any(generate_retry[row] for row in pending)
    model, processor = load_qwen3_vl(size) if needs_model else (None, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        for position, row in enumerate(pending, start=1):
            dataset_name = dataset_names[int(mixture["dataset_id"][row])]
            index = int(mixture["indices"][row])
            baseline = baselines[dataset_name][index]
            record = {
                "mixture_row": row,
                "dataset": dataset_name,
                "index": index,
                "head_confidence": float(probabilities[row]),
                "token_confidence": float(mixture["token_confidence"][row]),
                "flagged": bool(generate_retry[row]),
                "never_selected": False,
                "retry_all_selected": True,
                "head_threshold_selected": bool(threshold_selected[row]),
                "head_fraction_selected": bool(fraction_selected[row]),
                "token_fraction_selected": bool(token_selected[row]),
                "random_fraction_selected": bool(random_selected[row]),
                "question": baseline["question"],
                "reference_answer": baseline.get("reference_answer"),
                "baseline_prediction": baseline["prediction"],
                "baseline_correct": bool(baseline["correct"]),
                "baseline_score": float(baseline["score"]),
            }
            if generate_retry[row]:
                try:
                    example = datasets[dataset_name][index]
                    image = extract_image(example, raw_dir / dataset_name)
                    question = extract_question(example)
                    reference_answer, _ = extract_reference(example)
                    prompted = retry_question(question, baseline["prediction"]["answer"])
                    retry = generate_with_confidence(
                        model, processor, image, prompted, max_new_tokens
                    )
                    retry_correct, retry_score = score_prediction(
                        example, reference_answer, retry.answer, question
                    )
                    record.update(
                        retry_prediction=asdict(retry),
                        retry_correct=bool(retry_correct),
                        retry_score=float(retry_score),
                    )
                except Exception as exc:
                    record["retry_error"] = f"{type(exc).__name__}: {exc}"
            output_file.write(json.dumps(record, default=str) + "\n")
            output_file.flush()
            print(f"[{position}/{len(pending)}] {dataset_name}:{index} retry={bool(generate_retry[row])}")

    write_report(
        output_path,
        {
            "feature_path": str(feature_path),
            "head_path": str(head_path),
            "threshold": threshold,
            "review_fraction": fraction,
            "retry_all": retry_all,
            "random_seed": random_seed,
            "size": size,
        },
        {
            "baseline_no_retry": "never_selected",
            "retry_all": "retry_all_selected",
            "head_threshold": "head_threshold_selected",
            "head_lowest_fraction": "head_fraction_selected",
            "token_lowest_fraction": "token_fraction_selected",
            "random_matched_fraction": "random_fraction_selected",
        },
    )


def main() -> None:
    """Parse paths and selection policy for the prompted-retry experiment."""
    parser = argparse.ArgumentParser(description="Retry low-confidence mixture-test answers.")
    parser.add_argument("--features", type=Path, default=FEATURE_DIR / "mixture_test_2b.pt")
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", choices=("2b", "4b"), default="2b")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--review-fraction", type=float, default=0.2)
    parser.add_argument("--retry-all", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_prompted_retry(
        args.features,
        args.head,
        args.baseline_dir,
        args.raw_dir,
        args.output,
        args.size,
        args.threshold,
        args.review_fraction,
        args.retry_all,
        args.random_seed,
        args.max_new_tokens,
        args.limit,
    )


if __name__ == "__main__":
    main()
