# escalation.py
#
# Generates the "second opinion" answers for the escalation study (report 13). The detection step
# stays our trained correctness head; this script produces the answers of two new kinds of fix,
# which analyze_escalation.py then hands only to the questions the head flags:
#
#   --fix model     greedy answer of a larger Qwen3-VL (default 8B) on the full image and the same
#                   prompt as the 2B greedy answer in the candidate pools
#   --fix upscale   greedy answer of the same 2B model on the image enlarged by each --scales factor
#                   (Lanczos), capped at --max-pixels. The processor's own limit is about 16.7M pixels,
#                   so every image here already goes in at native size; enlarging it gives small text
#                   more image tokens, which is the only way to give the 2B model a sharper look.
#
# Every validation and test question is answered, not only the flagged ones, so the routing study can
# try any detector and any budget offline without new generation, and the "fix everything" upper
# reference is measured on the same questions.
#
# Output: results/escalation/<fix>/<split>/<dataset>_shard<i>of<n>.jsonl, one record per question with
# the answer(s), correctness, token confidence and image size. Resumes from what it already wrote.

import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

from experiments.approach1_latent_self_correction.baseline import (
    extract_image,
    extract_question,
    extract_reference,
    score_prediction,
)
from experiments.approach1_latent_self_correction.candidate_pools import PROMPT_SUFFIX, cached_split
from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES, dataset_dir, shard_range
from models.load_model import load_qwen3_vl
from models.run_inference import generate_with_confidence

OUT_DIR = Path(__file__).parent / "results" / "escalation"
MAX_PIXELS = 8_000_000  # about 7,800 image tokens for Qwen3-VL (32 x 32 pixels per merged token)


def upscaled_size(width: int, height: int, scale: float, max_pixels: int = MAX_PIXELS) -> tuple[int, int, float]:
    """Size of the image enlarged by `scale`, reduced so it stays under max_pixels, and never below native."""
    effective = min(scale, math.sqrt(max_pixels / (width * height)))
    effective = max(effective, 1.0)
    return round(width * effective), round(height * effective), effective


def upscale(image: Image.Image, scale: float, max_pixels: int = MAX_PIXELS) -> tuple[Image.Image, float]:
    """The enlarged image and the scale actually applied."""
    width, height, effective = upscaled_size(*image.size, scale, max_pixels)
    if effective == 1.0:
        return image, 1.0
    return image.resize((width, height), Image.LANCZOS), effective


def answer(model, processor, example, image, question, reference, max_new_tokens: int) -> dict:
    """Greedy answer with its correctness, token confidence and time taken."""
    start = time.time()
    result = generate_with_confidence(model, processor, image, question + PROMPT_SUFFIX, max_new_tokens)
    correct, answer_score = score_prediction(example, reference, result.answer, question)
    return {"answer": result.answer, "correct": bool(correct), "answer_score": float(answer_score),
            "token_confidence": result.confidence, "mean_log_probability": result.mean_log_probability,
            "tokens": len(result.token_probabilities), "seconds": round(time.time() - start, 3)}


def run(fix: str, size: str, scales: list[float], dataset: str, split: str, shard: int, num_shards: int,
        max_examples: int | None, max_new_tokens: int, max_pixels: int, output: Path) -> None:
    """Answer one shard of one dataset split with the chosen fix."""
    done = set()
    if output.exists():
        done = {json.loads(line)["index"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    data = cached_split(dataset, split)
    model, processor = load_qwen3_vl(size if fix == "model" else "2b")
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
            record = {"dataset": dataset, "split": split, "index": index, "fix": fix, "width": image.size[0], "height": image.size[1]}
            if fix == "model":
                record["model"] = size
                record["answer"] = answer(model, processor, example, image, question, reference, max_new_tokens)
            else:
                record["upscaled"] = {}
                for scale in scales:
                    enlarged, effective = upscale(image, scale, max_pixels)
                    result = answer(model, processor, example, enlarged, question, reference, max_new_tokens)
                    result["effective_scale"] = round(effective, 4)
                    record["upscaled"][f"{scale:g}"] = result
        except Exception as exc:
            record = {"dataset": dataset, "split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        with output.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, default=str) + "\n")
        print(f"[{fix} {dataset} {split} {position}/{len(indices)}] index {index}", flush=True)
    print(f"[done] wrote {output}")


def main() -> None:
    """Parse options."""
    parser = argparse.ArgumentParser(description="Second-opinion answers for the escalation study.")
    parser.add_argument("--fix", required=True, choices=("model", "upscale"))
    parser.add_argument("--size", default="8b", help="Model size for --fix model.")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.5, 2.0], help="Enlargement factors for --fix upscale.")
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None, help="Cap per shard, for a smoke test.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    name = f"model_{args.size}" if args.fix == "model" else "upscale"
    output = args.output or OUT_DIR / name / args.split / f"{args.dataset}_shard{args.shard}of{args.num_shards}.jsonl"
    run(args.fix, args.size, args.scales, args.dataset, args.split, args.shard, args.num_shards, args.max_examples,
        args.max_new_tokens, args.max_pixels, output)


if __name__ == "__main__":
    main()
