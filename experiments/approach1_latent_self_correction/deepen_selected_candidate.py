# deepen_selected_candidate.py
#
# Takes the unified-head winner from a completed best-of-five run, asks Qwen to
# reconsider it more deeply, scores the revised answer, and evaluates whether
# the second pass should replace the selected candidate.

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiments.approach1_latent_self_correction.correction_scale import cached_test_split, dataset_dir
from experiments.approach1_latent_self_correction.baseline import (
    extract_image,
    extract_question,
    extract_reference,
    score_prediction,
)
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from experiments.approach1_latent_self_correction.extract_q_features import answer_messages, encode_state
from experiments.approach1_latent_self_correction.online_prompted_retry import DATASET_NAMES, load_test_split
from experiments.approach1_latent_self_correction.prompted_retry import policy_summary
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages, generate_with_confidence

RESULTS_DIR = Path(__file__).parent / "results"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RESULTS_DIR / "mixture_test_2b_best_of_five_limit100.jsonl"
DEFAULT_OUTPUT = RESULTS_DIR / "mixture_test_2b_best_of_five_deep_limit100.jsonl"
DEFAULT_HEAD = REPO_ROOT / "trained_heads" / "approach1" / "winners" / "mixture" / "unified_new_seed1.pt"
MARGINS = (0.0, 0.05, 0.1, 0.2)
DEEP_PROMPT = (
    "{question}\nA candidate answer was: {answer}\n"
    "Think more deeply before answering. Re-examine the visual evidence, check the relevant "
    "labels, values, and relationships, and correct the candidate if needed.\n"
    "Respond with only the final answer, without explanation."
)


def add_selections(records: list[dict]) -> None:
    """Attach replacement decisions for automatic, head-reranked, and oracle policies."""
    for record in records:
        selected_score = record["selected_candidate"]["head_confidence"]
        deep_score = record["deep_answer"]["head_confidence"]
        record.update(
            keep_selected=False,
            deep_replace_all=True,
            deep_head_rerank=deep_score > selected_score,
            deep_oracle=bool(record["deep_answer"]["correct"] and not record["selected_candidate"]["correct"]),
        )
        for margin in MARGINS:
            key = str(margin).replace(".", "_")
            record[f"deep_margin_{key}"] = deep_score > selected_score + margin


def write_summary(records: list[dict], input_path: Path, output_path: Path) -> None:
    """Write pooled and per-dataset metrics for every second-pass policy."""
    policies = {
        "keep_best_of_five": "keep_selected",
        "deep_replace_all": "deep_replace_all",
        "deep_head_rerank": "deep_head_rerank",
        "deep_oracle": "deep_oracle",
    }
    for margin in MARGINS:
        key = str(margin).replace(".", "_")
        policies[f"deep_margin_{margin:.2f}"] = f"deep_margin_{key}"

    adapted = []
    for record in records:
        row = {
            **record,
            "baseline_correct": record["selected_candidate"]["correct"],
            "retry_correct": record["deep_answer"]["correct"],
        }
        adapted.append(row)
    summary = {
        "input": str(input_path),
        "examples": len(records),
        "prompt": DEEP_PROMPT,
        "policies": {name: policy_summary(adapted, key) for name, key in policies.items()},
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {output_path} and {summary_path}")


def run(input_path: Path, output_path: Path, head_path: Path, max_new_tokens: int, cached: bool = False, summarize: bool = True) -> None:
    """Run one deeper reconsideration pass for every best-of-five head winner in the input records."""
    source = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = [record for record in source if "candidates" in record]
    source_by_dataset = {
        name: {record["index"]: record for record in source if record["dataset"] == name}
        for name in DATASET_NAMES
    }
    existing = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()] if output_path.exists() else []
    completed = {(record["dataset"], record["index"]) for record in existing if "deep_answer" in record}
    head, spec, temperatures = load_head(head_path)
    model, processor = load_qwen3_vl("2b")
    records = list(existing)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASET_NAMES:
        source_rows = source_by_dataset[dataset_name]
        if not source_rows:
            continue
        materialize = max(source_rows) + 1
        split = cached_test_split(dataset_name) if cached else load_test_split(dataset_name, materialize_limit=materialize)
        for position, index in enumerate(sorted(source_rows), start=1):
            if (dataset_name, index) in completed:
                continue
            original = source_rows[index]
            chosen_index = original["selected_indices"]["head_best"]
            chosen = original["candidates"][chosen_index]
            example = split[index]
            try:
                image = extract_image(example, dataset_dir(dataset_name) if cached else Path("."))
                question = extract_question(example)
                reference, _ = extract_reference(example)
                original_prompt = f"{question}\nRespond with only the final answer, without explanation."
                prompt_state = encode_state(
                    model, processor, build_messages(image, original_prompt), add_generation_prompt=True
                )
                deep = generate_with_confidence(
                    model,
                    processor,
                    image,
                    DEEP_PROMPT.format(question=question, answer=chosen["answer"]),
                    max_new_tokens,
                )
                correct, answer_score = score_prediction(example, reference, deep.answer, question)
                deep_state = encode_state(
                    model, processor, answer_messages(image, original_prompt, deep.answer)
                )
                head_confidence = float(score(
                    head, spec, temperatures, prompt_state.unsqueeze(0), deep_state.unsqueeze(0)
                )["joint"][0])
                record = {
                    "dataset": dataset_name,
                    "index": index,
                    "question": question,
                    "reference_answer": reference,
                    "selected_candidate_index": chosen_index,
                    "selected_candidate": chosen,
                    "deep_answer": {
                        **asdict(deep),
                        "head_confidence": head_confidence,
                        "correct": bool(correct),
                        "answer_score": float(answer_score),
                    },
                }
            except Exception as exc:
                record = {"dataset": dataset_name, "index": index, "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            with output_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record, default=str) + "\n")
            print(f"[{dataset_name} {position}/{len(source_rows)}] wrote index {index}", flush=True)

    if not summarize:
        print(f"[done] wrote {output_path}")
        return
    successful = [record for record in records if "deep_answer" in record]
    add_selections(successful)
    errors = [record for record in records if "deep_answer" not in record]
    with output_path.open("w", encoding="utf-8") as destination:
        for record in successful + errors:
            destination.write(json.dumps(record, default=str) + "\n")
    write_summary(successful, input_path, output_path)


def main() -> None:
    """Parse completed best-of-five input and second-pass output paths."""
    parser = argparse.ArgumentParser(description="Reconsider the best-of-five selected answer.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cached-splits", action="store_true", help="Read the test splits cached by the Q-head pipeline instead of downloading them.")
    parser.add_argument("--no-summary", action="store_true", help="Write only the records, for sharded runs merged later.")
    args = parser.parse_args()
    run(args.input, args.output, args.head, args.max_new_tokens, args.cached_splits, not args.no_summary)


if __name__ == "__main__":
    main()
