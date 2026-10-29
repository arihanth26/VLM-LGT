# candidate_reranking.py
#
# Generates several sampled answers for each mixture-test question, scores every
# answer with the unified correctness head, compares selection baselines, and
# writes both machine-readable results and a self-contained HTML report.

import argparse
import html
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.baseline import (
    extract_image,
    extract_question,
    extract_reference,
    score_prediction,
)
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from experiments.approach1_latent_self_correction.extract_q_features import answer_messages, encode_state
from experiments.approach1_latent_self_correction.correction_scale import cached_test_split, dataset_dir, shard_range
from experiments.approach1_latent_self_correction.online_prompted_retry import DATASET_NAMES, load_test_split
from experiments.approach1_latent_self_correction.scoring import normalize_vqa_answer
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages, generate_with_confidence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEAD = REPO_ROOT / "trained_heads" / "approach1" / "winners" / "mixture" / "unified_new_seed1.pt"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "mixture_test_2b_best_of_five.jsonl"
POLICIES = ("greedy", "first_sample", "majority_vote", "token_best", "head_best", "oracle")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch before one sampled generation."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def majority_index(candidates: list[dict]) -> int:
    """Choose the most common normalized answer, breaking ties by head score."""
    normalized = [normalize_vqa_answer(candidate["answer"]) for candidate in candidates]
    counts = Counter(normalized)
    largest = max(counts.values())
    eligible = [index for index, answer in enumerate(normalized) if counts[answer] == largest]
    return max(eligible, key=lambda index: candidates[index]["head_confidence"])


def select_indices(record: dict) -> dict[str, int]:
    """Return the candidate selected by each policy, including the oracle bound."""
    candidates = record["candidates"]
    correct = [index for index, candidate in enumerate(candidates) if candidate["correct"]]
    return {
        "greedy": -1,
        "first_sample": 0,
        "majority_vote": majority_index(candidates),
        "token_best": max(range(len(candidates)), key=lambda index: candidates[index]["token_confidence"]),
        "head_best": max(range(len(candidates)), key=lambda index: candidates[index]["head_confidence"]),
        "oracle": correct[0] if correct else 0,
    }


def policy_metrics(records: list[dict], policy: str) -> dict:
    """Compute accuracy and answer diversity for one selection policy."""
    if not records:
        return {"examples": 0, "accuracy": 0.0, "correct": 0}
    correct = 0
    for record in records:
        if policy == "greedy":
            correct += bool(record["greedy"]["correct"])
        else:
            correct += bool(record["candidates"][record["selected_indices"][policy]]["correct"])
    return {"examples": len(records), "correct": correct, "accuracy": correct / len(records)}


def build_summary(records: list[dict], metadata: dict) -> dict:
    """Build pooled and per-dataset metrics for all candidate selectors."""
    return {
        **metadata,
        "policies": {
            policy: {
                "pooled": policy_metrics(records, policy),
                "per_dataset": {
                    name: policy_metrics([record for record in records if record["dataset"] == name], policy)
                    for name in DATASET_NAMES
                },
            }
            for policy in POLICIES
        },
        "candidate_diversity": {
            "mean_unique_answers": float(np.mean([
                len({normalize_vqa_answer(candidate["answer"]) for candidate in record["candidates"]})
                for record in records
            ])) if records else 0.0,
            "unanimous_fraction": float(np.mean([
                len({normalize_vqa_answer(candidate["answer"]) for candidate in record["candidates"]}) == 1
                for record in records
            ])) if records else 0.0,
        },
    }


def write_html(records: list[dict], summary: dict, output_path: Path) -> None:
    """Render aggregate metrics and informative selection cases as HTML."""
    labels = {
        "greedy": "Greedy baseline", "first_sample": "First sampled answer",
        "majority_vote": "Majority vote", "token_best": "Best token confidence",
        "head_best": "Best unified-head score", "oracle": "Oracle upper bound",
    }
    rows = []
    for policy in POLICIES:
        result = summary["policies"][policy]["pooled"]
        cells = [f"<td>{html.escape(labels[policy])}</td>", f"<td>{result['correct']}/{result['examples']}</td>", f"<td>{result['accuracy']:.1%}</td>"]
        for name in DATASET_NAMES:
            cells.append(f"<td>{summary['policies'][policy]['per_dataset'][name]['accuracy']:.1%}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    cases = []
    for record in records:
        head = record["candidates"][record["selected_indices"]["head_best"]]
        greedy = record["greedy"]
        if head["correct"] == greedy["correct"]:
            continue
        candidates = "<br>".join(
            f"{index + 1}. {html.escape(candidate['answer'])} "
            f"(head {candidate['head_confidence']:.3f}, token {candidate['token_confidence']:.3f}, "
            f"{'correct' if candidate['correct'] else 'wrong'})"
            for index, candidate in enumerate(record["candidates"])
        )
        cases.append(
            "<tr>"
            f"<td>{html.escape(record['dataset'])}:{record['index']}</td>"
            f"<td>{html.escape(record['question'])}</td>"
            f"<td>{html.escape(str(record['reference_answer']))}</td>"
            f"<td>{html.escape(greedy['answer'])} ({'correct' if greedy['correct'] else 'wrong'})</td>"
            f"<td>{candidates}</td><td>{html.escape(head['answer'])}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Best-of-five candidate reranking</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 20px;color:#18202a}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #ccd3da;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef2f5}}.metric{{display:inline-block;background:#eef2f5;padding:12px 18px;margin-right:12px;border-radius:8px}}</style>
</head><body><h1>Best-of-five candidate reranking</h1>
<p>Five answers are sampled at temperature {summary['temperature']} and top-p {summary['top_p']}. Each candidate is scored by the frozen unified correctness head.</p>
<div class="metric">Examples: {len(records)}</div><div class="metric">Mean unique answers: {summary['candidate_diversity']['mean_unique_answers']:.2f}</div><div class="metric">Unanimous: {summary['candidate_diversity']['unanimous_fraction']:.1%}</div>
<h2>Accuracy</h2><table><thead><tr><th>Policy</th><th>Correct</th><th>Pooled</th>{''.join(f'<th>{html.escape(name)}</th>' for name in DATASET_NAMES)}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Cases where head selection changes greedy correctness</h2><table><thead><tr><th>Example</th><th>Question</th><th>Reference</th><th>Greedy</th><th>Sampled candidates</th><th>Head selection</th></tr></thead><tbody>{''.join(cases) or '<tr><td colspan="6">No changed outcomes.</td></tr>'}</tbody></table>
</body></html>"""
    output_path.write_text(document, encoding="utf-8")


def run(
    head_path: Path,
    output_path: Path,
    candidates: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_new_tokens: int,
    limit: int | None,
    datasets: tuple | None = None,
    shard: int = 0,
    num_shards: int = 1,
    cached: bool = False,
    summarize: bool = True,
    max_examples: int | None = None,
) -> None:
    """Run best-of-N generation and confidence-based selection, on the whole test split or one shard of it."""
    if candidates < 2:
        raise ValueError("candidates must be at least 2")
    existing = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()] if output_path.exists() else []
    completed = {(record["dataset"], record["index"]) for record in existing if "candidates" in record}
    head, spec, temperatures = load_head(head_path)
    model, processor = load_qwen3_vl("2b")
    quotas = None
    if limit is not None:
        base, remainder = divmod(limit, len(DATASET_NAMES))
        quotas = {name: base + (position < remainder) for position, name in enumerate(DATASET_NAMES)}

    records = list(existing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for dataset_position, dataset_name in enumerate(DATASET_NAMES):
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
                reference, _ = extract_reference(example)
                prompt = f"{question}\nRespond with only the final answer, without explanation."
                prompt_state = encode_state(model, processor, build_messages(image, prompt), add_generation_prompt=True)
                seed_everything(seed + dataset_position * 1_000_000 + index * 100)
                greedy = generate_with_confidence(model, processor, image, prompt, max_new_tokens)
                greedy_correct, greedy_score = score_prediction(example, reference, greedy.answer, question)
                sampled = []
                for candidate_index in range(candidates):
                    seed_everything(seed + dataset_position * 1_000_000 + index * 100 + candidate_index + 1)
                    answer = generate_with_confidence(
                        model, processor, image, prompt, max_new_tokens,
                        do_sample=True, temperature=temperature, top_p=top_p,
                    )
                    correct, answer_score = score_prediction(example, reference, answer.answer, question)
                    state = encode_state(model, processor, answer_messages(image, prompt, answer.answer))
                    head_confidence = float(score(
                        head, spec, temperatures, prompt_state.unsqueeze(0), state.unsqueeze(0)
                    )["joint"][0])
                    sampled.append({
                        **asdict(answer), "token_confidence": answer.confidence,
                        "head_confidence": head_confidence, "correct": bool(correct),
                        "answer_score": float(answer_score),
                    })
                record = {
                    "dataset": dataset_name, "index": index, "question": question,
                    "reference_answer": reference,
                    "greedy": {**asdict(greedy), "correct": bool(greedy_correct), "answer_score": float(greedy_score)},
                    "candidates": sampled,
                }
                record["selected_indices"] = select_indices(record)
            except Exception as exc:
                record = {"dataset": dataset_name, "index": index, "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            with output_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record, default=str) + "\n")
            print(f"[{dataset_name} {index + 1 - indices.start}/{len(indices)}] wrote index {index}", flush=True)

    if not summarize:
        print(f"[done] wrote {output_path}")
        return
    successful = [record for record in records if "candidates" in record]
    metadata = {
        "head_path": str(head_path), "candidate_count": candidates,
        "temperature": temperature, "top_p": top_p, "seed": seed, "limit": limit,
    }
    summary = build_summary(successful, metadata)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_html(successful, summary, output_path.with_suffix(".html"))
    print(f"[done] wrote {output_path}, {summary_path}, and {output_path.with_suffix('.html')}")


def main() -> None:
    """Parse best-of-N candidate generation options."""
    parser = argparse.ArgumentParser(description="Generate and rerank sampled VLM answers.")
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None, help="Only these datasets.")
    parser.add_argument("--shard", type=int, default=0, help="Which block of each test split to run.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cached-splits", action="store_true", help="Read the test splits cached by the Q-head pipeline instead of downloading them.")
    parser.add_argument("--no-summary", action="store_true", help="Write only the records, for sharded runs merged later.")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap the examples per dataset and shard, for a smoke test.")
    args = parser.parse_args()
    run(args.head, args.output, args.candidates, args.temperature, args.top_p, args.seed, args.max_new_tokens, args.limit,
        None if args.datasets is None else tuple(args.datasets), args.shard, args.num_shards, args.cached_splits, not args.no_summary, args.max_examples)


if __name__ == "__main__":
    main()
