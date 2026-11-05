# candidate_pools.py
#
# Generates a pool of candidate answers per question for the candidate-search study: the
# greedy answer plus K sampled answers at one temperature, sampled in a single batched
# generate call. Selection policies for every k <= K are evaluated later on prefixes of the
# same pool (analyze_candidate_pools.py), so best-of-3, 5, 7 and 10 come from one run.
#
# The question embedding is held constant. Every head score in this file uses the cached
# prompt state from q_features/<dataset>_<split>_2b.pt (the exact tensor the heads were
# trained and tested on), never a recomputed one, so all candidates, temperatures and
# head-training targets see the same question representation. Only the answer half of the
# unified head changes between candidates.
#
# Two modes:
#   scored (val, test)   greedy + K samples, each distinct answer scored by the trained
#                        unified head and the linear probe (joint mode), plus the
#                        question-level pre-generation score of both
#   --labels-only (train) K samples and their correctness only, used as pass-rate
#                        targets for the pre-generation head sweep (sweep_pre_head.py)
#
# Output is one JSONL per shard; it resumes from what it already wrote.

import argparse
import json
from pathlib import Path

import torch

from experiments.approach1_latent_self_correction.baseline import (
    extract_image,
    extract_question,
    extract_reference,
    score_prediction,
)
from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES, dataset_dir, shard_range
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from experiments.approach1_latent_self_correction.extract_q_features import answer_messages, encode_state
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages, generate_with_confidence, sample_with_confidence

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = Path(__file__).parent / "q_features"
HEADS = REPO_ROOT / "trained_heads" / "approach1" / "winners" / "mixture"
DEFAULT_HEAD = HEADS / "unified_new_seed1.pt"
DEFAULT_PROBE = HEADS / "unified_new_linear_seed1.pt"
POOL_DIR = Path(__file__).parent / "results" / "pools"
PROMPT_SUFFIX = "\nRespond with only the final answer, without explanation."


def cached_split(name: str, split: str):
    """One split of a dataset, loaded from the cache written by the Q-head pipeline."""
    from datasets import load_from_disk

    return load_from_disk(str(dataset_dir(name)))[split]


def question_embeddings(name: str, split: str, size: str = "2b") -> dict[int, torch.Tensor]:
    """The cached prompt state of every question, by dataset index: the constant question embedding."""
    cached = torch.load(FEATURE_DIR / f"{name}_{split}_{size}.pt", map_location="cpu", weights_only=True)
    return {int(index): cached["prompt_features"][row] for row, index in enumerate(cached["indices"].tolist())}


def compact(result, correct: bool, answer_score: float, max_new_tokens: int) -> dict:
    """One candidate without its per-token probabilities, which the pool does not need."""
    return {
        "answer": result.answer,
        "token_confidence": result.confidence,
        "mean_log_probability": result.mean_log_probability,
        "tokens": len(result.token_probabilities),
        "capped": len(result.token_probabilities) >= max_new_tokens,
        "correct": bool(correct),
        "answer_score": float(answer_score),
    }


def question_seed(seed: int, dataset: str, index: int) -> int:
    """Sampling seed of one question; the same at every temperature, so pools differ only by temperature."""
    return seed + DATASET_NAMES.index(dataset) * 1_000_000 + index * 100


def run(
    dataset: str, split: str, output: Path, samples: int, temperature: float, top_p: float, seed: int,
    max_new_tokens: int, shard: int, num_shards: int, max_examples: int | None, labels_only: bool,
    head_path: Path, probe_path: Path,
) -> None:
    """Generate and (unless labels_only) score one shard of candidate pools."""
    done = set()
    if output.exists():
        done = {json.loads(line)["index"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    data = cached_split(dataset, split)
    embeddings = {} if labels_only else question_embeddings(dataset, split)
    heads = {} if labels_only else {"head": load_head(head_path), "probe": load_head(probe_path)}
    model, processor = load_qwen3_vl("2b")
    output.parent.mkdir(parents=True, exist_ok=True)
    indices = shard_range(len(data), shard, num_shards, max_examples)

    for position, index in enumerate(indices, start=1):
        if index in done:
            continue
        example = data[index]
        try:
            image = extract_image(example, dataset_dir(dataset))
            question = extract_question(example)
            reference, _ = extract_reference(example)
            prompt = question + PROMPT_SUFFIX
            torch.manual_seed(question_seed(seed, dataset, index))
            sampled = sample_with_confidence(model, processor, image, prompt, samples, temperature, top_p, max_new_tokens)
            candidates = []
            for result in sampled:
                correct, answer_score = score_prediction(example, reference, result.answer, question)
                candidates.append(compact(result, correct, answer_score, max_new_tokens))
            record = {"dataset": dataset, "split": split, "index": index, "temperature": temperature,
                      "top_p": top_p, "samples": candidates}
            if not labels_only:
                greedy = generate_with_confidence(model, processor, image, prompt, max_new_tokens)
                correct, answer_score = score_prediction(example, reference, greedy.answer, question)
                record["greedy"] = compact(greedy, correct, answer_score, max_new_tokens)
                question_state = embeddings[index].float().unsqueeze(0)
                record["pre_generation"] = {
                    name: float(score(head, spec, temperatures, question_state)["pre_generation"][0])
                    for name, (head, spec, temperatures) in heads.items()
                }
                # Each distinct answer string is encoded once; identical answers get identical scores.
                states = {}
                for candidate in [record["greedy"], *candidates]:
                    if candidate["answer"] not in states:
                        states[candidate["answer"]] = encode_state(
                            model, processor, answer_messages(image, prompt, candidate["answer"])
                        ).unsqueeze(0)
                    for name, (head, spec, temperatures) in heads.items():
                        candidate[f"{name}_confidence"] = float(
                            score(head, spec, temperatures, question_state, states[candidate["answer"]])["joint"][0]
                        )
                record["reference_answer"] = reference
                record["question"] = question
        except Exception as exc:
            record = {"dataset": dataset, "split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
        with output.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, default=str) + "\n")
        print(f"[{dataset} {split} T={temperature} {position}/{len(indices)}] index {index}", flush=True)
    print(f"[done] wrote {output}")


def check_embedding(dataset: str, split: str, count: int = 3) -> None:
    """Compare the cached question embedding with a freshly computed one on a few questions."""
    data = cached_split(dataset, split)
    embeddings = question_embeddings(dataset, split)
    model, processor = load_qwen3_vl("2b")
    for index in sorted(embeddings)[:count]:
        example = data[index]
        image = extract_image(example, dataset_dir(dataset))
        fresh = encode_state(model, processor, build_messages(image, extract_question(example) + PROMPT_SUFFIX),
                             add_generation_prompt=True)
        cached = embeddings[index].float()
        cosine = torch.nn.functional.cosine_similarity(fresh, cached, dim=0).item()
        print(f"[embedding] {dataset} {split} {index}: cosine {cosine:.6f}, max abs diff {(fresh - cached).abs().max():.4f}")


def main() -> None:
    """Parse pool-generation options."""
    parser = argparse.ArgumentParser(description="Generate candidate pools for the candidate-search study.")
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None, help="Cap per shard, for a smoke test.")
    parser.add_argument("--labels-only", action="store_true", help="Samples and correctness only (train pass rates).")
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check-embedding", action="store_true", help="Only compare cached and fresh question embeddings.")
    args = parser.parse_args()
    if args.check_embedding:
        check_embedding(args.dataset, args.split)
        return
    output = args.output or POOL_DIR / f"T{args.temperature:g}" / args.split / f"{args.dataset}_shard{args.shard}of{args.num_shards}.jsonl"
    run(args.dataset, args.split, output, args.samples, args.temperature, args.top_p, args.seed, args.max_new_tokens,
        args.shard, args.num_shards, args.max_examples, args.labels_only, args.head, args.probe)


if __name__ == "__main__":
    main()
