# measure_latency.py
#
# Wall-clock cost of every component of the correction pipeline, measured on one GPU, so cost can be
# reported in seconds as well as in model passes. For the first --per-dataset test questions of each
# dataset, each component runs once and is timed with CUDA synchronization:
#
#   greedy          2B short answer (64 tokens)              samples_10     10 sampled 2B answers, one batched call
#   reasoning       2B step-by-step answer (256/384 tokens)  reasoning_x3   3 sampled 2B reasoning answers, one call
#   grounding       2B box request + grounded crop answer + crop-with-context answer
#   zoom            one 2B answer on a corner crop            vcd            one contrastive-decoding 2B answer
#   encode          one candidate's state and probability     judge          one candidate's yes/no judgement
#   greedy_8b       8B short answer                          reasoning_8b   8B step-by-step answer
#
# The first question of each run is a warm-up and is not timed. Output: results/latency.json with the
# mean, median and 90th percentile seconds of each component, the GPU name, and the per-question raw times.

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.approach1_latent_self_correction.baseline import extract_image, extract_question
from experiments.approach1_latent_self_correction.candidate_pools import PROMPT_SUFFIX, cached_split
from experiments.approach1_latent_self_correction.correction_candidates import (
    VCD_ALPHAS,
    contrastive_answer,
    encode_candidate,
    zoom_crops,
)
from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES, dataset_dir
from experiments.approach1_latent_self_correction.correction_v2 import (
    CONTEXT_PROMPT,
    COT_PROMPT,
    GROUND_PROMPT,
    expand_box,
    generate_two_images,
    judge_probability,
    parse_box,
    upscale,
)
from experiments.approach1_latent_self_correction.correction_v3_cot import COT_TOKENS, DEFAULT_COT_TOKENS, sample_reasoning
from models.load_model import load_qwen3_vl
from models.run_inference import generate_with_confidence, sample_with_confidence

RESULTS = Path(__file__).parent / "results"


def timed(function, *args, **kwargs):
    """Run a function and return (result, seconds), with the GPU synchronized before and after."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def grounding(model, processor, image, question: str, prompt: str) -> None:
    """The three calls of the grounded-crop candidates: box request, crop answer, crop-with-context answer."""
    box_text = generate_with_confidence(model, processor, image, GROUND_PROMPT.format(question=question), 48).answer
    box = parse_box(box_text, image.width, image.height) or (0, 0, image.width, image.height)
    region = upscale(image.crop(expand_box(box, image.width, image.height)))
    generate_with_confidence(model, processor, region, prompt, 64)
    generate_two_images(model, processor, image, region, CONTEXT_PROMPT.format(prompt=prompt), 64)


def components_2b(model, processor, image, question: str, dataset: str) -> dict:
    """Every 2B component on one question; returns seconds per component."""
    prompt = question + PROMPT_SUFFIX
    budget = COT_TOKENS.get(dataset, DEFAULT_COT_TOKENS)
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    seconds = {}
    greedy, seconds["greedy"] = timed(generate_with_confidence, model, processor, image, prompt, 64)
    _, seconds["samples_10"] = timed(sample_with_confidence, model, processor, image, prompt, 10, 1.0, 0.95, 64)
    _, seconds["reasoning"] = timed(generate_with_confidence, model, processor, image, COT_PROMPT.format(question=question), budget)
    _, seconds["reasoning_x3"] = timed(sample_reasoning, model, processor, image, question, 3, 0.7, budget)
    _, seconds["grounding"] = timed(grounding, model, processor, image, question, prompt)
    _, seconds["zoom"] = timed(generate_with_confidence, model, processor, zoom_crops(image)[0], prompt, 64)
    _, seconds["vcd"] = timed(contrastive_answer, model, processor, image, prompt, VCD_ALPHAS[0], 3, 64)
    _, seconds["encode"] = timed(encode_candidate, model, processor, image, prompt, greedy.answer)
    _, seconds["judge"] = timed(judge_probability, model, processor, image, question, greedy.answer, yes_id, no_id)
    return seconds


def components_8b(model, processor, image, question: str, dataset: str) -> dict:
    """The 8B short answer and reasoning answer on one question."""
    budget = COT_TOKENS.get(dataset, DEFAULT_COT_TOKENS)
    seconds = {}
    _, seconds["greedy_8b"] = timed(generate_with_confidence, model, processor, image, question + PROMPT_SUFFIX, 64)
    _, seconds["reasoning_8b"] = timed(generate_with_confidence, model, processor, image, COT_PROMPT.format(question=question), budget)
    return seconds


def summarize(raw: dict[str, list[float]]) -> dict:
    """Mean, median and 90th percentile of each component."""
    out = {}
    for name, values in raw.items():
        ordered = sorted(values)
        out[name] = {"mean": statistics.mean(values), "median": statistics.median(values),
                     "p90": ordered[int(0.9 * (len(ordered) - 1))], "n": len(values)}
    return out


def main() -> None:
    """Time every component on the first questions of each test split, 2B then 8B."""
    parser = argparse.ArgumentParser(description="Wall-clock cost of every pipeline component.")
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--output", type=Path, default=RESULTS / "latency.json")
    parser.add_argument("--models", nargs="+", default=["2b", "8b"],
                        help="2b: every 2B component; 8b: 8B short and reasoning answers; any other key of models/model_config.py: its short answer.")
    args = parser.parse_args()
    questions = []
    for dataset in DATASET_NAMES:
        data = cached_split(dataset, "test")
        for index in range(args.per_dataset + 1):  # +1: the first is a warm-up
            example = data[index]
            questions.append((dataset, index, extract_image(example, dataset_dir(dataset)), extract_question(example)))
    raw: dict[str, list[float]] = {}
    per_question = []
    other = lambda size: (lambda model, processor, image, question, dataset: {f"greedy_{size}": timed(
        generate_with_confidence, model, processor, image, question + PROMPT_SUFFIX, 64)[1]})
    for size, run in [(s, {"2b": components_2b, "8b": components_8b}.get(s) or other(s)) for s in args.models]:
        model, processor = load_qwen3_vl(size)
        for position, (dataset, index, image, question) in enumerate(questions):
            with torch.no_grad():
                seconds = run(model, processor, image, question, dataset)
            if index == 0:
                continue  # warm-up
            per_question.append({"dataset": dataset, "index": index, "size": size, **seconds})
            for name, value in seconds.items():
                raw.setdefault(name, []).append(value)
            if position % 20 == 0:
                print(f"[{size}] {position}/{len(questions)}", flush=True)
        del model
        torch.cuda.empty_cache()
    result = {"gpu": torch.cuda.get_device_name(0), "per_dataset": args.per_dataset, "components": summarize(raw), "per_question": per_question}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"gpu": result["gpu"], **{k: round(v["mean"], 3) for k, v in result["components"].items()}}, indent=1))


if __name__ == "__main__":
    main()
