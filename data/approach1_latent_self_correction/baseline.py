# baseline.py
#
# Runs ordinary full-image Qwen3-VL inference over an Approach 1 dataset split.
# This establishes the unmodified-model baseline before latent interventions.

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from PIL import Image

from data.approach1_latent_self_correction.dataset_config import (
    OOD_DATASETS,
    QHEAD_DATASETS,
    TESTBED_DATASETS,
    TRAINING_DATASETS,
)
from data.approach1_latent_self_correction.report import write_markdown_report
from data.approach1_latent_self_correction.scoring import score_with_metric
from models.load_model import load_qwen3_vl
from models.run_inference import generate_with_confidence

RAW_DIR = Path(__file__).parent / "raw"
RESULTS_DIR = Path(__file__).parent / "results"
DATASET_GROUPS = {
    "training": TRAINING_DATASETS,
    "testbed": TESTBED_DATASETS,
    "ood": OOD_DATASETS,
    "qhead": QHEAD_DATASETS,
}


def first_present(example: dict, names: tuple[str, ...]):
    """Return the first present, non-null field without coercing arrays to bool."""
    for name in names:
        if name in example and example[name] is not None:
            return example[name]
    return None


def extract_image(example: dict, dataset_dir: Path) -> Image.Image:
    """Load an RGB image from common Hugging Face image representations."""
    value = first_present(example, ("image", "img", "image_path", "file_name"))
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path")
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = dataset_dir / path
        with Image.open(path) as source_image:
            return source_image.convert("RGB")
    raise ValueError("example has no supported image field")


def extract_question(example: dict) -> str:
    """Extract a VQA question or convert a referring expression into a prompt."""
    question = first_present(example, ("question", "query", "prompt"))
    if question is not None:
        return str(question)
    expression = first_present(example, ("sentence", "expression", "phrase", "text"))
    if expression is not None:
        return f"Which object or region is described by: {expression}?"
    raise ValueError("example has no supported question or referring-expression field")


def extract_reference(example: dict):
    """Extract available answer and bounding-box references."""
    answer = first_present(example, ("answer", "answers", "label", "target"))
    if isinstance(answer, (list, tuple)) and len(answer) == 1:
        answer = answer[0]
    bbox = first_present(example, ("bbox", "box", "bounding_box"))
    return answer, bbox


def numeric_interpretations(value: str) -> list[float]:
    """Parse numeric answers while allowing percent and proportion conventions."""
    text = str(value).strip().replace(",", "")
    try:
        if text.endswith("%"):
            number = float(text[:-1])
            return [number, number / 100.0]
        return [float(text)]
    except ValueError:
        return []


def relaxed_correctness(
    reference, prediction: str, question: str = "", tolerance: float = 0.05
) -> bool:
    """Score numeric answers with tolerance, but require exact years and normalized text."""
    reference_text = str(reference).strip().lower()
    prediction_text = str(prediction).strip().lower()
    reference_numbers = numeric_interpretations(reference_text)
    prediction_numbers = numeric_interpretations(prediction_text)
    if reference_numbers and prediction_numbers:
        if "year" in question.lower() or "when" in question.lower():
            return prediction_text.rstrip(".") == reference_text.rstrip(".")
        for reference_number in reference_numbers:
            for prediction_number in prediction_numbers:
                if reference_number == 0 and prediction_number == 0:
                    return True
                if reference_number != 0:
                    relative_error = abs(prediction_number - reference_number) / abs(reference_number)
                    if relative_error <= tolerance:
                        return True
        return False
    return prediction_text.rstrip(".") == reference_text.rstrip(".")


def score_prediction(example: dict, reference_answer, prediction: str, question: str) -> tuple[bool, float]:
    """Score with the dataset's own metric when it names one, else with relaxed ChartQA matching."""
    if example.get("metric"):
        return score_with_metric(example["metric"], list(example["answers"]), prediction)
    correct = relaxed_correctness(reference_answer, prediction, question=question)
    return correct, float(correct)


def run_baseline(
    dataset_name: str,
    group: str,
    split: str,
    size: str,
    output_path: Path,
    max_new_tokens: int,
    limit: int | None = None,
) -> None:
    """Run one full-image generation for every requested example in a dataset split."""
    if dataset_name not in DATASET_GROUPS[group]:
        choices = ", ".join(DATASET_GROUPS[group])
        raise ValueError(f"unknown {group} dataset '{dataset_name}'. Available: {choices}")

    dataset_dir = RAW_DIR / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset not found at {dataset_dir}; download it first")

    loaded = load_from_disk(str(dataset_dir))
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise ValueError(f"split '{split}' unavailable. Available: {', '.join(loaded.keys())}")
        split_data = loaded[split]
    else:
        split_data = loaded

    total = len(split_data) if limit is None else min(limit, len(split_data))
    completed_indices = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    completed_indices.add(json.loads(line)["index"])
    pending_indices = [index for index in range(total) if index not in completed_indices]
    if not pending_indices:
        write_markdown_report(output_path, output_path.with_suffix(".md"))
        print(f"[done] all {total} requested examples already exist")
        return

    model, processor = load_qwen3_vl(size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        for position, index in enumerate(pending_indices, start=1):
            example = split_data[index]
            try:
                image = extract_image(example, dataset_dir)
                question = extract_question(example)
                reference_answer, reference_bbox = extract_reference(example)
                inference_question = (
                    f"{question}\nRespond with only the final answer, without explanation."
                )
                prediction = generate_with_confidence(
                    model, processor, image, inference_question, max_new_tokens
                )
                correct, score = score_prediction(
                    example, reference_answer, prediction.answer, question
                )
                record = {
                    "index": index,
                    "dataset": dataset_name,
                    "split": split,
                    "question": question,
                    "reference_answer": reference_answer,
                    "reference_bbox": reference_bbox,
                    "prediction": asdict(prediction),
                    "correct": correct,
                    "score": score,
                }
            except Exception as exc:
                record = {
                    "index": index,
                    "dataset": dataset_name,
                    "split": split,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            output_file.write(json.dumps(record, default=str) + "\n")
            output_file.flush()
            print(f"[{position}/{len(pending_indices)}] wrote dataset index {index}")

    write_markdown_report(output_path, output_path.with_suffix(".md"))


def main() -> None:
    """Parse baseline options and evaluate the unmodified model."""
    parser = argparse.ArgumentParser(description="Run a full-image dataset baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--group", choices=DATASET_GROUPS, default="ood")
    parser.add_argument("--split", default="test")
    parser.add_argument("--size", choices=["2b", "4b"], default="2b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    output_path = args.output or RESULTS_DIR / f"{args.dataset}_{args.split}_{args.size}_baseline.jsonl"
    run_baseline(
        dataset_name=args.dataset,
        group=args.group,
        split=args.split,
        size=args.size,
        output_path=output_path,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
