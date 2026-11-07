# correction_candidates.py
#
# Builds the inputs of the learned correction step: new candidate answers from correction
# strategies, and the hidden state of every distinct candidate a question has, so a ranker can
# be trained to pick the right one (train_candidate_ranker.py).
#
# Correction strategies (each gives one greedy answer):
#   vcd_<alpha>   visual contrastive decoding: at each token, the log-probabilities under the real
#                 image are pushed away from those under a noised copy of it,
#                 (1 + alpha) log p(y | image) - alpha log p(y | noised image),
#                 restricted to tokens the real image finds plausible (p >= beta * max p). Tokens
#                 that the noised image would also produce are language priors, not evidence.
#   zoom_<i>      the same question on one of four overlapping crops (60% of each side, at the four
#                 corners) upscaled 2x, so small text and marks get more visual tokens
# Candidates kept from the pools: the greedy answer and every sample at T=1.0 and T=1.3 for val and
# test, and the greedy answer plus the 8 samples at T=1.0 for train.
#
# Every distinct answer string is then encoded once in the same context: the full image, the
# original prompt, the answer. That gives its answer state, a teacher-forced answer probability
# (the same number for a candidate whatever produced it), and the head and probe scores computed
# with the constant cached question embedding.
#
# Outputs per shard: <name>.jsonl (the correction candidates, like candidate_pools.py records) and
# <name>.pt (answer strings and states of every distinct candidate, per question).

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from experiments.approach1_latent_self_correction.baseline import (
    extract_image,
    extract_question,
    extract_reference,
    score_prediction,
)
from experiments.approach1_latent_self_correction.candidate_pools import (
    DEFAULT_HEAD,
    DEFAULT_PROBE,
    POOL_DIR,
    PROMPT_SUFFIX,
    cached_split,
    question_embeddings,
)
from experiments.approach1_latent_self_correction.correction_scale import dataset_dir, shard_range
from experiments.approach1_latent_self_correction.corruption import add_noise
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from experiments.approach1_latent_self_correction.extract_q_features import answer_messages
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages, generate_with_confidence

HERE = Path(__file__).parent
OUT_DIR = HERE / "results" / "correction"
VCD_ALPHAS = (0.5, 1.0)
VCD_BETA = 0.1
NOISE_STRENGTH = 0.15
CROPS = ((0.0, 0.0), (0.4, 0.0), (0.0, 0.4), (0.4, 0.4))
CROP_FRACTION = 0.6
MAX_ZOOM_SIDE = 2048


class ContrastiveLogits(LogitsProcessor):
    """Visual contrastive decoding: contrast each step's next-token distribution with a noised-image run."""

    def __init__(self, model, corrupt_inputs: dict, prompt_length: int, alpha: float, beta: float):
        self.model, self.corrupt, self.prompt_length = model, corrupt_inputs, prompt_length
        self.alpha, self.beta = alpha, beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_length:]
        inputs = dict(self.corrupt)
        inputs["input_ids"] = torch.cat([self.corrupt["input_ids"], generated], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        # Generated tokens are text, so their multimodal token type is 0.
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in inputs:
                inputs[key] = torch.cat([self.corrupt[key], torch.zeros_like(generated)], dim=1)
        with torch.no_grad():
            corrupt_logits = self.model(**inputs, use_cache=False).logits[:, -1].float()
        clean = torch.log_softmax(scores.float(), dim=-1)
        noised = torch.log_softmax(corrupt_logits, dim=-1)
        combined = (1 + self.alpha) * clean - self.alpha * noised
        implausible = clean < clean.max(dim=-1, keepdim=True).values + torch.log(torch.tensor(self.beta))
        return combined.masked_fill(implausible, float("-inf")).to(scores.dtype)


def chat_inputs(processor, image: Image.Image, prompt: str, device) -> dict:
    """Tokenized single-turn chat with the generation prompt."""
    return processor.apply_chat_template(
        build_messages(image, prompt), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)


def contrastive_answer(model, processor, image: Image.Image, prompt: str, alpha: float, seed: int, max_new_tokens: int) -> str:
    """Greedy decoding with visual contrastive logits against a noised copy of the image."""
    clean = chat_inputs(processor, image, prompt, model.device)
    corrupt = chat_inputs(processor, add_noise(image, seed, NOISE_STRENGTH), prompt, model.device)
    processor_list = LogitsProcessorList([ContrastiveLogits(model, corrupt, clean["input_ids"].shape[1], alpha, VCD_BETA)])
    with torch.no_grad():
        output = model.generate(**clean, max_new_tokens=max_new_tokens, do_sample=False, logits_processor=processor_list)
    return processor.decode(output[0, clean["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def zoom_crops(image: Image.Image) -> list[Image.Image]:
    """Four overlapping corner crops, each upscaled 2x (capped), for reading small detail."""
    image = image.convert("RGB")
    width, height = image.size
    crops = []
    for left, top in CROPS:
        box = (int(left * width), int(top * height), int((left + CROP_FRACTION) * width), int((top + CROP_FRACTION) * height))
        crop = image.crop(box)
        scale = min(2.0, MAX_ZOOM_SIDE / max(crop.size))
        crops.append(crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS))
    return crops


def encode_candidate(model, processor, image: Image.Image, prompt: str, answer: str) -> tuple[torch.Tensor, float, int]:
    """Last-token answer state and teacher-forced geometric-mean probability of the answer tokens, in one pass."""
    prefix = processor.apply_chat_template(build_messages(image, prompt), tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    full = processor.apply_chat_template(answer_messages(image, prompt, answer), tokenize=True, add_generation_prompt=False,
                                         return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**full, output_hidden_states=True, use_cache=False, return_dict=True)
    ids = full["input_ids"][0]
    end = int(full["attention_mask"][0].sum().item())
    start = prefix["input_ids"].shape[1]
    # Answer tokens and the end-of-turn token; the template's trailing newline is left out.
    stop = max(start + 1, end - 1)
    log_probs = torch.log_softmax(outputs.logits[0, start - 1:stop - 1].float(), dim=-1)
    token_log_probs = log_probs.gather(1, ids[start:stop].unsqueeze(1)).squeeze(1)
    confidence = float(token_log_probs.mean().exp()) if len(token_log_probs) else 0.0
    return outputs.hidden_states[-1][0, end - 1].detach().float().cpu(), confidence, int(stop - start)


def pool_records(split: str, dataset: str, temperatures: tuple[str, ...]) -> dict[str, dict[int, dict]]:
    """Existing pools of one dataset split, by temperature folder and index."""
    found = {}
    for temperature in temperatures:
        rows = {}
        for path in sorted((POOL_DIR / f"T{temperature}" / split).glob(f"{dataset}_shard*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if "samples" in record:
                        rows[record["index"]] = record
        found[temperature] = rows
    return found


def train_greedy(dataset: str) -> dict[int, dict]:
    """Greedy answers and correctness of the train split, from the baseline run."""
    rows = {}
    for line in (HERE / "results" / f"{dataset}_train_2b_baseline.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "prediction" in record:
                rows[record["index"]] = {"answer": record["prediction"]["answer"], "correct": bool(record["correct"])}
    return rows


def run(dataset: str, split: str, shard: int, num_shards: int, max_examples: int | None, corrections: bool,
        max_new_tokens: int, output: Path, corrections_limit: int | None = None) -> None:
    """Generate correction candidates and encode every distinct candidate of one shard."""
    data = cached_split(dataset, split)
    embeddings = question_embeddings(dataset, split)
    heads = {"head": load_head(DEFAULT_HEAD), "probe": load_head(DEFAULT_PROBE)}
    temperatures = ("1",) if split == "train" else ("1", "1.3")
    pools = pool_records(split, dataset, temperatures)
    greedy_train = train_greedy(dataset) if split == "train" else {}
    model, processor = load_qwen3_vl("2b")
    output.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if output.with_suffix(".jsonl").exists():
        done = {json.loads(l)["index"] for l in output.with_suffix(".jsonl").read_text().splitlines() if l.strip()}
    saved = torch.load(output.with_suffix(".pt"), weights_only=False) if output.with_suffix(".pt").exists() else {"questions": {}}

    indices = [i for i in shard_range(len(data), shard, num_shards, max_examples) if i in pools[temperatures[0]]]
    for position, index in enumerate(indices, start=1):
        if index in done:
            continue
        example = data[index]
        try:
            image = extract_image(example, dataset_dir(dataset))
            question = extract_question(example)
            reference, _ = extract_reference(example)
            prompt = question + PROMPT_SUFFIX
            question_state = embeddings[index].float().unsqueeze(0)
            record = {"dataset": dataset, "split": split, "index": index, "corrections": []}
            if corrections and (corrections_limit is None or position <= corrections_limit):
                seed = index * 7 + 3
                generated = [(f"vcd_{alpha:g}", contrastive_answer(model, processor, image, prompt, alpha, seed, max_new_tokens))
                             for alpha in VCD_ALPHAS]
                for number, crop in enumerate(zoom_crops(image)):
                    generated.append((f"zoom_{number}", generate_with_confidence(model, processor, crop, prompt, max_new_tokens).answer))
                for source, answer in generated:
                    correct, answer_score = score_prediction(example, reference, answer, question)
                    record["corrections"].append({"source": source, "answer": answer, "correct": bool(correct),
                                                  "answer_score": float(answer_score)})
            answers = []
            if split == "train":
                answers.append(greedy_train[index]["answer"])
            for temperature in temperatures:
                pool = pools[temperature].get(index)
                if pool is not None:
                    if "greedy" in pool:
                        answers.append(pool["greedy"]["answer"])
                    answers += [c["answer"] for c in pool["samples"]]
            answers += [c["answer"] for c in record["corrections"]]
            distinct = list(dict.fromkeys(answers))
            states, encoded = [], {}
            for answer in distinct:
                state, confidence, tokens = encode_candidate(model, processor, image, prompt, answer)
                head_scores = {name: float(score(h, spec, temps, question_state, state.unsqueeze(0))["joint"][0])
                               for name, (h, spec, temps) in heads.items()}
                correct, _ = score_prediction(example, reference, answer, question)
                encoded[answer] = {"tf_confidence": confidence, "tokens": tokens, "correct": bool(correct), **{f"{k}_confidence": v for k, v in head_scores.items()}}
                states.append(state.to(torch.float16))
            for candidate in record["corrections"]:
                candidate.update({k: v for k, v in encoded[candidate["answer"]].items() if k != "correct"})
            record["encoded"] = encoded
            saved["questions"][index] = {"answers": distinct, "states": torch.stack(states)}
        except Exception as exc:
            record = {"dataset": dataset, "split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
        with output.with_suffix(".jsonl").open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, default=str) + "\n")
        if position % 200 == 0:
            torch.save(saved, output.with_suffix(".pt"))
        print(f"[{dataset} {split} {position}/{len(indices)}] index {index}", flush=True)
    torch.save(saved, output.with_suffix(".pt"))
    print(f"[done] wrote {output.with_suffix('.jsonl')} and {output.with_suffix('.pt')}")


def main() -> None:
    """Parse shard options."""
    parser = argparse.ArgumentParser(description="Correction candidates and candidate states for the learned ranker.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None, help="Cap per shard (smoke test, or the train subset that gets corrections).")
    parser.add_argument("--no-corrections", action="store_true", help="Only encode the existing pool candidates.")
    parser.add_argument("--corrections-limit", type=int, default=None,
                        help="Generate corrections only for the first N questions of the shard; encode the rest (train subset).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    tag = "states" if args.no_corrections else "full"
    output = args.output or OUT_DIR / args.split / f"{args.dataset}_{tag}_shard{args.shard}of{args.num_shards}"
    run(args.dataset, args.split, args.shard, args.num_shards, args.max_examples, not args.no_corrections, args.max_new_tokens,
        output, args.corrections_limit)


if __name__ == "__main__":
    main()
