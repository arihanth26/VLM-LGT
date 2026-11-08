# correction_v2.py
#
# Second round of correction candidates and a new ranker feature, on top of correction_candidates.py:
#   ground       the model is asked for the bounding box of the region that answers the question
#                (Qwen3-VL gives boxes on a 0-1000 grid); that region, with a margin, is cropped,
#                upscaled 2x, and the question is asked on it. Unlike the four fixed corner crops,
#                the crop follows the evidence.
#   ground_ctx   the same region given as a second image next to the full image, so the model reads
#                the detail without losing the context
#   cot          step-by-step reasoning (up to COT_TOKENS tokens) ending with "Answer: ...", from which
#                the final answer is extracted; aimed at errors of reasoning over read values (charts)
#   judge        for every distinct candidate of the question, the model is shown the image, the
#                question and the candidate and asked whether it is correct; the feature is
#                P(Yes) / (P(Yes) + P(No)) of its next token. One forward pass per candidate.
# For training questions without the first-round corrections (contrastive decoding, corner crops),
# --old-corrections generates them as well, so the ranker can be trained with every source on many
# more questions than before.
#
# New candidates are encoded like every other (full image, original prompt): answer state,
# teacher-forced answer probability, head and probe scores with the constant cached question embedding.
# Outputs per shard: <name>.jsonl and <name>.pt (states of answers not already encoded).

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

from experiments.approach1_latent_self_correction.baseline import (
    extract_image,
    extract_question,
    extract_reference,
    score_prediction,
)
from experiments.approach1_latent_self_correction.candidate_pools import (
    DEFAULT_HEAD,
    DEFAULT_PROBE,
    PROMPT_SUFFIX,
    cached_split,
    question_embeddings,
)
from experiments.approach1_latent_self_correction.correction_candidates import (
    VCD_ALPHAS,
    contrastive_answer,
    encode_candidate,
    zoom_crops,
)
from experiments.approach1_latent_self_correction.correction_scale import dataset_dir, shard_range
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from models.load_model import load_qwen3_vl
from models.run_inference import generate_with_confidence

HERE = Path(__file__).parent
FIRST_ROUND = HERE / "results" / "correction"
OUT_DIR = HERE / "results" / "correction_v2"
COT_TOKENS = 256
BOX_MARGIN = 0.15
MIN_BOX = 0.2
MAX_SIDE = 2048
GROUND_PROMPT = ('Locate the region of the image that contains the information needed to answer this question: "{question}" '
                 'Output only its bounding box as JSON: {{"bbox_2d": [x1, y1, x2, y2]}}.')
CONTEXT_PROMPT = "The second image is an enlarged view of the relevant region of the first image.\n{prompt}"
COT_PROMPT = ("{question}\nThink step by step in at most three short steps, then write the final answer on the last line "
              "in the form 'Answer: <answer>'.")
JUDGE_PROMPT = "{question}\nProposed answer: {answer}\nLook at the image carefully. Is the proposed answer correct? Reply with Yes or No."


NUMBER = r"\s*(-?\d+(?:\.\d+)?)\s*"
BOX_LIST = re.compile(r"\[" + ",".join([NUMBER] * 4) + r"\]")


def parse_box(text: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    """The first bracketed list of four numbers in the reply as a pixel box, read on the 0-1000 grid when it fits.

    Only a bracketed list counts: the key "bbox_2d" itself contains a digit, so taking the first four
    numbers of the reply would read the 2 of the key as a coordinate.
    """
    match = BOX_LIST.search(text)
    if match is None:
        return None
    numbers = [float(n) for n in match.groups()]
    x1, y1, x2, y2 = numbers
    if max(numbers) <= 1000:
        x1, x2, y1, y2 = x1 / 1000 * width, x2 / 1000 * width, y1 / 1000 * height, y2 / 1000 * height
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def expand_box(box: tuple, width: int, height: int) -> tuple[int, int, int, int]:
    """Add a margin and enforce a minimum size, so the crop keeps nearby labels."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    w2, h2 = max(w * (1 + 2 * BOX_MARGIN), MIN_BOX * width), max(h * (1 + 2 * BOX_MARGIN), MIN_BOX * height)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = max(0, int(cx - w2 / 2)), max(0, int(cy - h2 / 2))
    return left, top, min(width, int(left + w2)), min(height, int(top + h2))


def upscale(image: Image.Image) -> Image.Image:
    """Upscale a crop 2x, capped at MAX_SIDE on the longest side."""
    scale = min(2.0, MAX_SIDE / max(image.size))
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def generate_two_images(model, processor, first: Image.Image, second: Image.Image, prompt: str, max_new_tokens: int) -> str:
    """Greedy answer with two images in one user turn."""
    messages = [{"role": "user", "content": [{"type": "image", "image": first}, {"type": "image", "image": second},
                                             {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract_final(text: str) -> str | None:
    """The answer after the last 'Answer:'; None when the reasoning never reached one (for example cut off)."""
    matches = re.findall(r"answer\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if not matches:
        return None
    answer = matches[-1].strip().strip("*").strip()
    return answer or None


def judge_probability(model, processor, image: Image.Image, question: str, answer: str, yes_id: int, no_id: int) -> float:
    """The model's own P(Yes) that a candidate answer is correct, normalized over Yes and No."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": JUDGE_PROMPT.format(question=question, answer=answer)}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
    pair = torch.softmax(torch.stack([logits[yes_id], logits[no_id]]), dim=0)
    return float(pair[0])


def first_round(split: str, dataset: str) -> dict[int, dict]:
    """First-round records (encoded answers and corrections) of one dataset split, by index."""
    records = {}
    for path in sorted((FIRST_ROUND / split).glob(f"{dataset}_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if "encoded" in record and (record["index"] not in records or record.get("corrections")):
                    records[record["index"]] = record
    return records


def run(dataset: str, split: str, shard: int, num_shards: int, max_examples: int | None, old_corrections: bool,
        max_new_tokens: int, output: Path) -> None:
    """Generate the second-round candidates and the judge feature for one shard."""
    data = cached_split(dataset, split)
    embeddings = question_embeddings(dataset, split)
    previous = first_round(split, dataset)
    heads = {"head": load_head(DEFAULT_HEAD), "probe": load_head(DEFAULT_PROBE)}
    model, processor = load_qwen3_vl("2b")
    tokenizer = processor.tokenizer
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.with_suffix(".jsonl").exists():
        done = {json.loads(l)["index"] for l in output.with_suffix(".jsonl").read_text().splitlines() if l.strip()}
    saved = torch.load(output.with_suffix(".pt"), weights_only=False) if output.with_suffix(".pt").exists() else {"questions": {}}

    indices = [i for i in shard_range(len(data), shard, num_shards, max_examples) if i in previous]
    for position, index in enumerate(indices, start=1):
        if index in done:
            continue
        example = data[index]
        try:
            image = extract_image(example, dataset_dir(dataset)).convert("RGB")
            question = extract_question(example)
            reference, _ = extract_reference(example)
            prompt = question + PROMPT_SUFFIX
            prior = previous[index]
            proposals = []
            if old_corrections and not prior.get("corrections"):
                seed = index * 7 + 3
                proposals += [(f"vcd_{alpha:g}", contrastive_answer(model, processor, image, prompt, alpha, seed, max_new_tokens))
                              for alpha in VCD_ALPHAS]
                proposals += [(f"zoom_{number}", generate_with_confidence(model, processor, crop, prompt, max_new_tokens).answer)
                              for number, crop in enumerate(zoom_crops(image))]
            box_text = generate_with_confidence(model, processor, image, GROUND_PROMPT.format(question=question), 48).answer
            box = parse_box(box_text, image.width, image.height)
            if box is not None:
                region = upscale(image.crop(expand_box(box, image.width, image.height)))
                proposals.append(("ground", generate_with_confidence(model, processor, region, prompt, max_new_tokens).answer))
                proposals.append(("ground_ctx", generate_two_images(model, processor, image, region, CONTEXT_PROMPT.format(prompt=prompt), max_new_tokens)))
            reasoning = generate_with_confidence(model, processor, image, COT_PROMPT.format(question=question), COT_TOKENS)
            final = extract_final(reasoning.answer)
            if final is not None:
                proposals.append(("cot", final))

            record = {"dataset": dataset, "split": split, "index": index, "box_reply": box_text, "box": box,
                      "cot_text": reasoning.answer, "cot_capped": len(reasoning.token_probabilities) >= COT_TOKENS, "corrections": []}
            question_state = embeddings[index].float().unsqueeze(0)
            known = set(prior["encoded"])
            encoded, states, new_answers = {}, [], []
            for source, answer in proposals:
                correct, answer_score = score_prediction(example, reference, answer, question)
                record["corrections"].append({"source": source, "answer": answer, "correct": bool(correct), "answer_score": float(answer_score)})
                if answer not in known and answer not in encoded:
                    state, confidence, tokens = encode_candidate(model, processor, image, prompt, answer)
                    encoded[answer] = {"tf_confidence": confidence, "tokens": tokens, "correct": bool(correct),
                                       **{f"{name}_confidence": float(score(h, spec, temps, question_state, state.unsqueeze(0))["joint"][0])
                                          for name, (h, spec, temps) in heads.items()}}
                    states.append(state.to(torch.float16))
                    new_answers.append(answer)
            record["encoded"] = encoded
            record["judge"] = {answer: judge_probability(model, processor, image, question, answer, yes_id, no_id)
                               for answer in [*prior["encoded"], *new_answers]}
            if states:
                saved["questions"][index] = {"answers": new_answers, "states": torch.stack(states)}
        except Exception as exc:
            record = {"dataset": dataset, "split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
        with output.with_suffix(".jsonl").open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, default=str) + "\n")
        if position % 100 == 0:
            torch.save(saved, output.with_suffix(".pt"))
        print(f"[{dataset} {split} {position}/{len(indices)}] index {index}", flush=True)
    torch.save(saved, output.with_suffix(".pt"))
    print(f"[done] wrote {output.with_suffix('.jsonl')} and {output.with_suffix('.pt')}")


def main() -> None:
    """Parse shard options."""
    parser = argparse.ArgumentParser(description="Second-round correction candidates and the self-judge feature.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None, help="First N questions of the shard (train subset, smoke test).")
    parser.add_argument("--old-corrections", action="store_true", help="Also generate contrastive decoding and corner crops where missing.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or OUT_DIR / args.split / f"{args.dataset}_shard{args.shard}of{args.num_shards}"
    run(args.dataset, args.split, args.shard, args.num_shards, args.max_examples, args.old_corrections, args.max_new_tokens, output)


if __name__ == "__main__":
    main()
