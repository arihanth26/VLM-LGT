# extend_pools.py
#
# Adds sampled answers to the existing validation and test pools so best-of-k can be measured past
# k = 10 (k = 11, 13, 15). The extra samples use a different seed from the first ten, at the same
# temperature and top-p, so samples 11 onwards are independent draws from the same distribution and
# pool k is still the greedy answer plus the first k samples. Each new distinct answer is encoded in
# the same context as every other candidate (full image, original prompt): answer state,
# teacher-forced answer probability, and head and probe scores on the constant cached question
# embedding, so the trained candidate ranker can score it unchanged.
#
# Output per dataset, split and temperature: results/correction_extra/<split>/<dataset>_T<t>.jsonl
# (the extra samples and the encoded features of new answers) and a .pt with their states.

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
from experiments.approach1_latent_self_correction.candidate_pools import (
    DEFAULT_HEAD,
    DEFAULT_PROBE,
    POOL_DIR,
    PROMPT_SUFFIX,
    cached_split,
    compact,
    question_embeddings,
    question_seed,
)
from experiments.approach1_latent_self_correction.correction_candidates import encode_candidate
from experiments.approach1_latent_self_correction.correction_scale import dataset_dir, shard_range
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from models.load_model import load_qwen3_vl
from models.run_inference import sample_with_confidence

HERE = Path(__file__).parent
CORRECTION_DIR = HERE / "results" / "correction"
OUT_DIR = HERE / "results" / "correction_extra"
SEED_OFFSET = 50  # question_seed spaces questions 100 apart, so +50 never meets another question's seed


def known_answers(dataset: str, split: str) -> dict[int, set]:
    """Distinct answers already encoded for each question by correction_candidates.py."""
    known: dict[int, set] = {}
    for path in sorted((CORRECTION_DIR / split).glob(f"{dataset}_*.pt")):
        for index, entry in torch.load(path, weights_only=False)["questions"].items():
            known.setdefault(index, set()).update(entry["answers"])
    return known


def run(dataset: str, split: str, temperature: str, extra: int, top_p: float, seed: int, max_new_tokens: int,
        shard: int, num_shards: int, max_examples: int | None, output: Path) -> None:
    """Sample extra answers for one dataset split at one temperature and encode the new ones."""
    pools = {}
    for path in sorted((POOL_DIR / f"T{temperature}" / split).glob(f"{dataset}_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if "samples" in record:
                    pools[record["index"]] = record
    data = cached_split(dataset, split)
    embeddings = question_embeddings(dataset, split)
    known = known_answers(dataset, split)
    heads = {"head": load_head(DEFAULT_HEAD), "probe": load_head(DEFAULT_PROBE)}
    model, processor = load_qwen3_vl("2b")
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.with_suffix(".jsonl").exists():
        done = {json.loads(l)["index"] for l in output.with_suffix(".jsonl").read_text().splitlines() if l.strip()}
    saved = torch.load(output.with_suffix(".pt"), weights_only=False) if output.with_suffix(".pt").exists() else {"questions": {}}

    indices = [i for i in shard_range(len(data), shard, num_shards, max_examples) if i in pools]
    for position, index in enumerate(indices, start=1):
        if index in done:
            continue
        example = data[index]
        try:
            image = extract_image(example, dataset_dir(dataset))
            question = extract_question(example)
            reference, _ = extract_reference(example)
            prompt = question + PROMPT_SUFFIX
            torch.manual_seed(question_seed(seed, dataset, index) + SEED_OFFSET)
            sampled = sample_with_confidence(model, processor, image, prompt, extra, float(temperature), top_p, max_new_tokens)
            samples = []
            for result in sampled:
                correct, answer_score = score_prediction(example, reference, result.answer, question)
                samples.append(compact(result, correct, answer_score, max_new_tokens))
            question_state = embeddings[index].float().unsqueeze(0)
            new_answers = [a for a in dict.fromkeys(s["answer"] for s in samples) if a not in known.get(index, set())]
            encoded, states = {}, []
            for answer in new_answers:
                state, confidence, tokens = encode_candidate(model, processor, image, prompt, answer)
                correct, _ = score_prediction(example, reference, answer, question)
                encoded[answer] = {"tf_confidence": confidence, "tokens": tokens, "correct": bool(correct),
                                   **{f"{name}_confidence": float(score(h, spec, temps, question_state, state.unsqueeze(0))["joint"][0])
                                      for name, (h, spec, temps) in heads.items()}}
                states.append(state.to(torch.float16))
            if states:
                saved["questions"][index] = {"answers": new_answers, "states": torch.stack(states)}
            record = {"dataset": dataset, "split": split, "index": index, "temperature": temperature, "samples": samples, "encoded": encoded}
        except Exception as exc:
            record = {"dataset": dataset, "split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
        with output.with_suffix(".jsonl").open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, default=str) + "\n")
        if position % 250 == 0:
            torch.save(saved, output.with_suffix(".pt"))
        print(f"[{dataset} {split} T={temperature} {position}/{len(indices)}] index {index}", flush=True)
    torch.save(saved, output.with_suffix(".pt"))
    print(f"[done] wrote {output.with_suffix('.jsonl')} and {output.with_suffix('.pt')}")


def main() -> None:
    """Parse options."""
    parser = argparse.ArgumentParser(description="Extend the candidate pools past ten samples.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--temperature", required=True, help="Pool folder name without the T, for example 1 or 1.3.")
    parser.add_argument("--extra", type=int, default=6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    # No dot in the stem (T1.3 is written T1p3), so with_suffix cannot cut the temperature off the name.
    tag = str(args.temperature).replace(".", "p")
    output = args.output or OUT_DIR / args.split / f"{args.dataset}_T{tag}_shard{args.shard}of{args.num_shards}"
    run(args.dataset, args.split, args.temperature, args.extra, args.top_p, args.seed, args.max_new_tokens,
        args.shard, args.num_shards, args.max_examples, output)


if __name__ == "__main__":
    main()
