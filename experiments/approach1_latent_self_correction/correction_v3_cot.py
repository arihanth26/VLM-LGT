# correction_v3_cot.py
#
# Several step-by-step reasoning answers per question, on top of the second round (correction_v2.py).
#
# Why: the single reasoning answer ("cot") is the strongest correction source (ChartQA +4.16 alone), but
# it is right only 82% of the time and about 3% of the time never reaches an "Answer:" line. Sampling a
# few reasoning paths gives the ranker more diverse candidates, and their agreement is a signal (the v3
# ranker's from_cot feature is the share of offered reasoning answers that agree with a candidate).
#
# What, per question already covered by correction_v2 (same train subset, val and test):
#   cot_s0 .. cot_s{N-1}   N reasoning answers sampled at TEMPERATURE in one generate call
#                          (num_return_sequences), with a larger budget for charts
#   encoded / states       every new answer encoded like the other candidates (full image, original
#                          prompt): answer state, teacher-forced probability, head and probe scores
#   judge                  the model's own P(Yes) for every new answer, as in correction_v2
# Outputs per shard: results/correction_v3/<split>/<dataset>_shard<k>of<n>.jsonl and .pt, in the
# correction_v2 record format, so train_ranker_v3.py can merge them. Resumable.

import argparse
import json
import re
from pathlib import Path

import torch

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
from experiments.approach1_latent_self_correction.correction_candidates import encode_candidate
from experiments.approach1_latent_self_correction.correction_scale import dataset_dir
from experiments.approach1_latent_self_correction.correction_v2 import COT_PROMPT, extract_final, judge_probability
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages

HERE = Path(__file__).parent
V2_DIR = HERE / "results" / "correction_v2"
OUT_DIR = HERE / "results" / "correction_v3"
SAMPLES = 3
TEMPERATURE = 0.7
TOP_P = 0.95
COT_TOKENS = {"chartqa_full": 384}
DEFAULT_COT_TOKENS = 256


def cot_sources(n: int = SAMPLES) -> tuple[str, ...]:
    """Correction source names of the sampled reasoning answers."""
    return tuple(f"cot_s{i}" for i in range(n))


def v2_indices(dataset: str, split: str) -> list[int]:
    """Indices of the questions correction_v2 covered without error, in order."""
    indices = set()
    for path in sorted((V2_DIR / split).glob(f"{dataset}_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "judge" in record:
                    indices.add(record["index"])
    return sorted(indices)


def v2_known_answers(dataset: str, split: str) -> dict[int, set]:
    """Per question, answers correction_v2 (or the first round, via its judge keys) already encoded."""
    known = {}
    for path in sorted((V2_DIR / split).glob(f"{dataset}_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "judge" in record:
                    known[record["index"]] = set(record["judge"]) | set(record.get("encoded", {}))
    return known


def sample_reasoning(model, processor, image, question: str, n: int, temperature: float, max_new_tokens: int) -> list[tuple[str, bool]]:
    """N sampled reasoning texts in one call, each with whether it hit the token limit."""
    inputs = processor.apply_chat_template(build_messages(image, COT_PROMPT.format(question=question)), tokenize=True,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                top_p=TOP_P, num_return_sequences=n)
    eos = {processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id} - {None}
    texts = []
    for row in output[:, inputs["input_ids"].shape[1]:]:
        ids = row.tolist()
        capped = len(ids) >= max_new_tokens and not any(t in eos for t in ids)
        texts.append((processor.decode(row, skip_special_tokens=True).strip(), capped))
    return texts


def shard_of(indices: list[int], shard: int, num_shards: int, max_examples: int | None) -> list[int]:
    """Every num_shards-th question starting at shard, optionally the first max_examples of them."""
    part = indices[shard::num_shards]
    return part[:max_examples] if max_examples else part


def run(dataset: str, split: str, shard: int, num_shards: int, max_examples: int | None, samples: int,
        temperature: float, output: Path) -> None:
    """Generate the sampled reasoning answers, their encodings and judge scores for one shard."""
    indices = shard_of(v2_indices(dataset, split), shard, num_shards, max_examples)
    known = v2_known_answers(dataset, split)
    data = cached_split(dataset, split)
    embeddings = question_embeddings(dataset, split)
    heads = {"head": load_head(DEFAULT_HEAD), "probe": load_head(DEFAULT_PROBE)}
    model, processor = load_qwen3_vl("2b")
    tokenizer = processor.tokenizer
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    budget = COT_TOKENS.get(dataset, DEFAULT_COT_TOKENS)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.with_suffix(".jsonl").exists():
        done = {json.loads(l)["index"] for l in output.with_suffix(".jsonl").read_text().splitlines() if l.strip()}
    saved = torch.load(output.with_suffix(".pt"), weights_only=False) if output.with_suffix(".pt").exists() else {"questions": {}}
    print(f"[start] {dataset} {split} shard {shard}/{num_shards}: {len(indices)} questions, {len(done)} done, budget {budget}", flush=True)

    for position, index in enumerate(indices, start=1):
        if index in done:
            continue
        example = data[index]
        try:
            image = extract_image(example, dataset_dir(dataset)).convert("RGB")
            question = extract_question(example)
            reference, _ = extract_reference(example)
            prompt = question + PROMPT_SUFFIX
            texts = sample_reasoning(model, processor, image, question, samples, temperature, budget)
            record = {"dataset": dataset, "split": split, "index": index, "cot_texts": [t for t, _ in texts],
                      "cot_capped": [c for _, c in texts], "corrections": []}
            question_state = embeddings[index].float().unsqueeze(0)
            seen = known.get(index, set())
            encoded, states, new_answers = {}, [], []
            for source, (text, _) in zip(cot_sources(samples), texts):
                answer = extract_final(text)
                if answer is None:
                    continue
                correct, answer_score = score_prediction(example, reference, answer, question)
                record["corrections"].append({"source": source, "answer": answer, "correct": bool(correct), "answer_score": float(answer_score)})
                if answer not in seen and answer not in encoded:
                    state, confidence, tokens = encode_candidate(model, processor, image, prompt, answer)
                    encoded[answer] = {"tf_confidence": confidence, "tokens": tokens, "correct": bool(correct),
                                       **{f"{name}_confidence": float(score(h, spec, temps, question_state, state.unsqueeze(0))["joint"][0])
                                          for name, (h, spec, temps) in heads.items()}}
                    states.append(state.to(torch.float16))
                    new_answers.append(answer)
            record["encoded"] = encoded
            record["judge"] = {a: judge_probability(model, processor, image, question, a, yes_id, no_id) for a in new_answers}
            if states:
                saved["questions"][index] = {"answers": new_answers, "states": torch.stack(states)}
        except Exception as exc:
            record = {"dataset": dataset, "split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
        with output.with_suffix(".jsonl").open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, default=str) + "\n")
        if position % 100 == 0:
            torch.save(saved, output.with_suffix(".pt"))
        if position % 25 == 0:
            print(f"[{dataset} {split} {position}/{len(indices)}] index {index}", flush=True)
    torch.save(saved, output.with_suffix(".pt"))
    print(f"[done] wrote {output.with_suffix('.jsonl')} and {output.with_suffix('.pt')}")


def main() -> None:
    """Parse shard options."""
    parser = argparse.ArgumentParser(description="Several sampled reasoning answers per question (third correction round).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None, help="First N questions of the shard (smoke test).")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or OUT_DIR / args.split / f"{args.dataset}_shard{args.shard}of{args.num_shards}"
    run(args.dataset, args.split, args.shard, args.num_shards, args.max_examples, args.samples, args.temperature, output)


if __name__ == "__main__":
    main()
