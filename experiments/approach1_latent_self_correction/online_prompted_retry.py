# online_prompted_retry.py
#
# Runs the prompted-retry benchmark without cached baselines or feature files.
# Dataset caches live in node-local temporary storage, while only compact JSONL
# results are written to the repository.
#
# Every answer is scored by two exported heads in the same pass: the trained
# unified head (--head) and the best linear probe of the same sweep (--probe),
# so the two can be compared on identical generations. The summary also reports
# per-dataset detection AUROC and error AUPRC of the token, head and probe scores.

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from data.approach1_latent_self_correction.build_qhead_splits import (
    finalize,
    normalize_docvqa,
    normalize_scienceqa,
    normalize_textvqa,
    split_by_group,
    take,
)
from data.approach1_latent_self_correction.dataset_config import QHEAD_DATASETS, TRAINING_DATASETS
from experiments.approach1_latent_self_correction.correction_scale import cached_test_split, dataset_dir, shard_range
from experiments.approach1_latent_self_correction.baseline import (
    extract_image,
    extract_question,
    extract_reference,
    score_prediction,
)
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from experiments.approach1_latent_self_correction.extract_q_features import answer_messages, encode_state
from experiments.approach1_latent_self_correction.prompted_retry import (
    confidence_mask,
    policy_summary,
    retry_question,
)
from experiments.approach1_latent_self_correction.train_q_heads import average_precision, roc_auc
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages, generate_with_confidence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEAD = REPO_ROOT / "trained_heads" / "approach1" / "winners" / "mixture" / "unified_new_seed1.pt"
DEFAULT_PROBE = REPO_ROOT / "trained_heads" / "approach1" / "winners" / "mixture" / "unified_new_linear_seed1.pt"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "mixture_test_2b_prompted_retry_reranked.jsonl"
DATASET_NAMES = ("chartqa_full", "textvqa", "docvqa", "scienceqa_img")
RERANK_MARGINS = (0.0, 0.05, 0.1, 0.2)


def load_test_split(
    name: str,
    test_size: int = 2500,
    val_size: int = 1500,
    seed: int = 42,
    max_side: int = 1536,
    materialize_limit: int | None = None,
):
    """Download and recreate one test split using the original evaluation protocol."""
    if name == "chartqa_full":
        entry = TRAINING_DATASETS[name]
        return (
            load_dataset(entry.hf_repo_id, name=entry.hf_config, split="test")
            if entry.hf_config
            else load_dataset(entry.hf_repo_id, split="test")
        )

    entry = QHEAD_DATASETS[name]
    if name == "textvqa":
        source = load_dataset(entry.hf_repo_id, split="validation")
        held_out = split_by_group(
            [str(value) for value in source["image_id"]],
            {"val": val_size, "test": test_size},
            seed,
        )
        indices = held_out["test"][:materialize_limit]
        return finalize(source.select(indices), normalize_textvqa, max_side)
    if name == "docvqa":
        source = load_dataset(entry.hf_repo_id, split="validation")
        held_out = split_by_group(
            [str(value) for value in source["ucsf_document_id"]],
            {"val": val_size, "test": test_size},
            seed,
        )
        indices = held_out["test"][:materialize_limit]
        return finalize(source.select(indices), normalize_docvqa, max_side)
    if name == "scienceqa_img":
        source = load_dataset(entry.hf_repo_id, split="test")
        with_image = source.filter(lambda example: example["image"] is not None)
        selected = take(with_image, test_size, seed)
        if materialize_limit is not None:
            selected = selected.select(range(min(materialize_limit, len(selected))))
        return finalize(selected, normalize_scienceqa, max_side)
    raise ValueError(f"unknown mixture dataset '{name}'")


def add_scorer_selections(records: list[dict], scorer: str, threshold: float, fraction: float) -> None:
    """Attach the threshold, lowest-fraction, and reranking decisions of one learned scorer."""
    scores = np.asarray([record[f"{scorer}_confidence"] for record in records])
    below = confidence_mask(scores, threshold, None)
    lowest = confidence_mask(scores, threshold, fraction)
    for index, record in enumerate(records):
        baseline_score = record[f"{scorer}_confidence"]
        retry_score = record[f"retry_{scorer}_confidence"]
        record.update({
            f"{scorer}_threshold_selected": bool(below[index]),
            f"{scorer}_fraction_selected": bool(lowest[index]),
            f"{scorer}_rerank_selected": retry_score > baseline_score,
        })
        for margin in RERANK_MARGINS:
            name = str(margin).replace(".", "_")
            record[f"{scorer}_gated_margin_{name}_selected"] = bool(
                baseline_score < threshold and retry_score > baseline_score + margin
            )


def add_policy_selections(records: list[dict], threshold: float, fraction: float, random_seed: int) -> None:
    """Attach paired policy decisions after all online confidence scores exist."""
    head_scores = np.asarray([record["head_confidence"] for record in records])
    token_scores = np.asarray([record["token_confidence"] for record in records])
    head_fraction = confidence_mask(head_scores, threshold, fraction)
    token_fraction = confidence_mask(token_scores, threshold, fraction)
    random_fraction = np.zeros(len(records), dtype=bool)
    count = int(head_fraction.sum())
    if count:
        generator = np.random.default_rng(random_seed)
        random_fraction[generator.choice(len(records), count, replace=False)] = True
    for index, record in enumerate(records):
        record.update(
            never_selected=False,
            retry_all_selected=True,
            token_fraction_selected=bool(token_fraction[index]),
            random_fraction_selected=bool(random_fraction[index]),
            oracle_selected=bool(record["retry_correct"] and not record["baseline_correct"]),
        )
    add_scorer_selections(records, "head", threshold, fraction)
    if records and all("probe_confidence" in record for record in records):
        add_scorer_selections(records, "probe", threshold, fraction)


def detection_metrics(records: list[dict]) -> dict:
    """AUROC and error AUPRC of every available score against baseline correctness."""
    labels = np.asarray([record["baseline_correct"] for record in records], dtype=float)
    result = {"examples": len(records), "accuracy": float(labels.mean()) if len(labels) else 0.0}
    for scorer in ("token", "head", "probe"):
        if not records or any(f"{scorer}_confidence" not in record for record in records):
            continue
        scores = np.asarray([record[f"{scorer}_confidence"] for record in records], dtype=float)
        result[scorer] = {
            "auroc": roc_auc(labels, scores),
            "error_auprc": average_precision(1 - labels, 1 - scores),
        }
    return result


def detection_summary(records: list[dict]) -> dict:
    """Detection metrics pooled and for each dataset."""
    names = [name for name in DATASET_NAMES if any(record["dataset"] == name for record in records)]
    summary = {name: detection_metrics([record for record in records if record["dataset"] == name]) for name in names}
    summary["pooled"] = detection_metrics(records)
    return summary


def write_outputs(output_path: Path, records: list[dict], metadata: dict) -> None:
    """Persist annotated records and pooled plus per-dataset policy metrics."""
    with output_path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, default=str) + "\n")
    policies = {
        "baseline_no_retry": "never_selected",
        "retry_all": "retry_all_selected",
        "head_threshold": "head_threshold_selected",
        "head_lowest_fraction": "head_fraction_selected",
        "token_lowest_fraction": "token_fraction_selected",
        "random_matched_fraction": "random_fraction_selected",
        "head_rerank": "head_rerank_selected",
        "oracle_upper_bound": "oracle_selected",
    }
    for margin in RERANK_MARGINS:
        name = str(margin).replace(".", "_")
        policies[f"head_gated_margin_{margin:.2f}"] = f"head_gated_margin_{name}_selected"
    if records and "probe_threshold_selected" in records[0]:
        policies.update({
            "probe_threshold": "probe_threshold_selected",
            "probe_lowest_fraction": "probe_fraction_selected",
            "probe_rerank": "probe_rerank_selected",
        })
        for margin in RERANK_MARGINS:
            name = str(margin).replace(".", "_")
            policies[f"probe_gated_margin_{margin:.2f}"] = f"probe_gated_margin_{name}_selected"
    summary = {
        **metadata,
        "online": True,
        "policies": {name: policy_summary(records, key) for name, key in policies.items()},
        "detection": detection_summary(records),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {output_path} and {summary_path}")


def score_pair(heads: dict, prompt_state, answer_state) -> dict[str, float]:
    """Joint-mode probability of every loaded head for one prompt and answer state."""
    return {
        name: float(score(head, spec, temperatures, prompt_state.unsqueeze(0), answer_state.unsqueeze(0))["joint"][0])
        for name, (head, spec, temperatures) in heads.items()
    }


def run_online(
    head_path: Path,
    output_path: Path,
    threshold: float,
    review_fraction: float,
    random_seed: int,
    max_new_tokens: int,
    limit: int | None,
    datasets: tuple | None = None,
    shard: int = 0,
    num_shards: int = 1,
    cached: bool = False,
    summarize: bool = True,
    max_examples: int | None = None,
    probe_path: Path | None = DEFAULT_PROBE,
) -> None:
    """Generate, score, and retry the mixture test examples in one model pass, on the whole test split or one shard."""
    existing = []
    if output_path.exists():
        existing = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {(record["dataset"], record["index"]) for record in existing}

    heads = {"head": load_head(head_path)}
    if probe_path is not None:
        heads["probe"] = load_head(probe_path)
    model, processor = load_qwen3_vl("2b")
    quotas = None
    if limit is not None:
        base, remainder = divmod(limit, len(DATASET_NAMES))
        quotas = {name: base + (position < remainder) for position, name in enumerate(DATASET_NAMES)}

    records = list(existing)
    for dataset_name in DATASET_NAMES:
        if datasets is not None and dataset_name not in datasets:
            continue
        quota = None if quotas is None else quotas[dataset_name]
        split = cached_test_split(dataset_name) if cached else load_test_split(dataset_name, materialize_limit=quota)
        count = len(split) if quota is None else min(len(split), quota)
        indices = shard_range(count, shard, num_shards, max_examples)
        for index in indices:
            if (dataset_name, index) in completed:
                continue
            example = split[index]
            try:
                image = extract_image(example, dataset_dir(dataset_name) if cached else Path("."))
                question = extract_question(example)
                reference_answer, _ = extract_reference(example)
                baseline_prompt = f"{question}\nRespond with only the final answer, without explanation."
                baseline = generate_with_confidence(model, processor, image, baseline_prompt, max_new_tokens)
                baseline_correct, baseline_score = score_prediction(
                    example, reference_answer, baseline.answer, question
                )
                prompt_state = encode_state(
                    model, processor, build_messages(image, baseline_prompt), add_generation_prompt=True
                )
                answer_state = encode_state(
                    model, processor, answer_messages(image, baseline_prompt, baseline.answer)
                )
                baseline_scores = score_pair(heads, prompt_state, answer_state)
                retry = generate_with_confidence(
                    model, processor, image, retry_question(question, baseline.answer), max_new_tokens
                )
                retry_correct, retry_score = score_prediction(
                    example, reference_answer, retry.answer, question
                )
                retry_state = encode_state(
                    model,
                    processor,
                    answer_messages(image, baseline_prompt, retry.answer),
                )
                retry_scores = score_pair(heads, prompt_state, retry_state)
                record = {
                    "dataset": dataset_name,
                    "index": index,
                    "question": question,
                    "reference_answer": reference_answer,
                    **{f"{name}_confidence": value for name, value in baseline_scores.items()},
                    "token_confidence": baseline.confidence,
                    "baseline_prediction": asdict(baseline),
                    "baseline_correct": bool(baseline_correct),
                    "baseline_score": float(baseline_score),
                    "retry_prediction": asdict(retry),
                    **{f"retry_{name}_confidence": value for name, value in retry_scores.items()},
                    "retry_correct": bool(retry_correct),
                    "retry_score": float(retry_score),
                }
            except Exception as exc:
                record = {"dataset": dataset_name, "index": index, "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record, default=str) + "\n")
            print(f"[{dataset_name} {index + 1 - indices.start}/{len(indices)}] wrote index {index}", flush=True)

    if not summarize:
        print(f"[done] wrote {output_path}")
        return
    successful = [record for record in records if "baseline_correct" in record]
    add_policy_selections(successful, threshold, review_fraction, random_seed)
    errors = [record for record in records if "baseline_correct" not in record]
    write_outputs(
        output_path,
        successful + errors,
        {
            "head_path": str(head_path),
            "probe_path": None if probe_path is None else str(probe_path),
            "datasets": None if datasets is None else list(datasets),
            "shard": shard,
            "num_shards": num_shards,
            "threshold": threshold,
            "review_fraction": review_fraction,
            "random_seed": random_seed,
            "limit": limit,
        },
    )


def main() -> None:
    """Parse the storage-light online benchmark options."""
    parser = argparse.ArgumentParser(description="Run the mixture retry benchmark online.")
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--review-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None, help="Only these datasets.")
    parser.add_argument("--shard", type=int, default=0, help="Which block of each test split to run.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cached-splits", action="store_true", help="Read the test splits cached by the Q-head pipeline instead of downloading them.")
    parser.add_argument("--no-summary", action="store_true", help="Write only the records, for sharded runs merged later.")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap the examples per dataset and shard, for a smoke test.")
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE, help="Linear probe scored beside the head.")
    parser.add_argument("--no-probe", action="store_true", help="Score with the head only.")
    args = parser.parse_args()
    run_online(
        args.head,
        args.output,
        args.threshold,
        args.review_fraction,
        args.random_seed,
        args.max_new_tokens,
        args.limit,
        None if args.datasets is None else tuple(args.datasets),
        args.shard,
        args.num_shards,
        args.cached_splits,
        not args.no_summary,
        args.max_examples,
        None if args.no_probe else args.probe,
    )


if __name__ == "__main__":
    main()
