# escalation_candidates.py
#
# Turns the escalation answers (escalation.py: Qwen3-VL-8B on the full image, and 2B on the image
# enlarged 1.5x and 2x) into candidates for the v3 ranker, so the ranker can choose among the 2B
# corrections and the 8B answer instead of the gate sending a question to one fix or the other.
#
# Sources:  esc_8b   the 8B greedy answer
#           up1.5    the 2B answer on the image enlarged 1.5x
#           up2      the 2B answer on the image enlarged 2x
#           esc_internvl / esc_idefics   the greedy answers of InternVL3-8B and Idefics3-8B (other model families;
#                    escalation.py --size internvl3_8b / idefics3_8b), with --out-dir correction_v6
#           cot_8b   the 8B step-by-step reasoning answer (escalation.py --fix reasoning); not in the default
#                    set, run with --sources cot_8b --out-dir correction_v5 so the first three stay as they are
# Every answer not already encoded for its question (by the first round, correction_v2 or the sampled
# reasoning of correction_v3) is encoded exactly like the other candidates, by the 2B model on the full
# image and the original prompt: answer state, teacher-forced probability, head and probe scores with
# the constant cached question embedding, and the 2B model's own P(Yes) judge score. Nothing about the
# 8B model enters the features except the answer text, so the ranker's inputs stay on one scale.
#
# Base model: VLMLGT_MODEL (correction_scale.py) selects the model that encodes and judges the answers and the result
# folders (results/base_<size>/ for 4B); the escalation answers themselves are always read from results/escalation/.
# The enlarged-image answers come from the 2B model, so they are only offered with the 2B base model.
# Questions: those correction_v2 covered (train: the ranker's 12,000; all of val and test).
# Output per shard: results/correction_v4/<split>/<dataset>_shard<k>of<n>.jsonl and .pt, in the
# correction_v3 record format, merged by train_ranker_v3.py --with-escalation. Resumable.

import argparse
import json
from pathlib import Path

import torch

from experiments.approach1_latent_self_correction.baseline import extract_image, extract_question
from experiments.approach1_latent_self_correction.candidate_pools import (
    DEFAULT_HEAD,
    DEFAULT_PROBE,
    PROMPT_SUFFIX,
    cached_split,
    question_embeddings,
)
from experiments.approach1_latent_self_correction.correction_candidates import encode_candidate
from experiments.approach1_latent_self_correction.correction_scale import ALL_DATASETS, MODEL_SIZE, RESULTS_ROOT, SPLITS, dataset_dir
from experiments.approach1_latent_self_correction.correction_v2 import judge_probability
from experiments.approach1_latent_self_correction.correction_v3_cot import shard_of, v2_indices, v2_known_answers
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from models.load_model import load_qwen3_vl

HERE = Path(__file__).parent
ESCALATION = HERE / "results" / "escalation"
V3_DIR = RESULTS_ROOT / "correction_v3"
OUT_DIR = RESULTS_ROOT / "correction_v4"
SOURCES = ("esc_8b", "up1.5", "up2")
FAMILY_FOLDERS = {"esc_internvl": "model_internvl3_8b", "esc_idefics": "model_idefics3_8b"}
ALL_SOURCES = SOURCES + ("cot_8b",) + tuple(FAMILY_FOLDERS)


def escalation_answers(dataset: str, split: str) -> dict[int, dict[str, dict]]:
    """Per question, the answer dict of each escalation source."""
    out: dict[int, dict[str, dict]] = {}
    for path in sorted((ESCALATION / "model_8b" / split).glob(f"{dataset}_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else {}
            if "answer" in r:
                out.setdefault(r["index"], {})["esc_8b"] = r["answer"]
    for path in sorted((ESCALATION / "upscale" / split).glob(f"{dataset}_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else {}
            for scale, result in r.get("upscaled", {}).items():
                out.setdefault(r["index"], {})[f"up{scale}"] = result
    for source, folder in FAMILY_FOLDERS.items():
        for path in sorted((ESCALATION / folder / split).glob(f"{dataset}_shard*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line) if line.strip() else {}
                if "answer" in r:
                    out.setdefault(r["index"], {})[source] = r["answer"]
    for path in sorted((ESCALATION / "reasoning_8b" / split).glob(f"{dataset}_shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("reasoning", {}).get("answer") is not None:  # no "Answer:" line: no candidate
                out.setdefault(r["index"], {})["cot_8b"] = r["reasoning"]
    return out


def known_answers(dataset: str, split: str) -> dict[int, set]:
    """Answers already encoded per question: the first round, correction_v2 (its judge keys), correction_v3 and
    correction_v4 (so a later --out-dir round never encodes an answer twice)."""
    known = v2_known_answers(dataset, split)
    later = [RESULTS_ROOT / f"correction_v{n}" for n in (4, 5, 6)]  # every escalation round, so no answer is encoded twice
    for path in sorted((V3_DIR / split).glob(f"{dataset}_shard*.jsonl")) + [p for d in later for p in sorted((d / split).glob(f"{dataset}_shard*.jsonl"))]:
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else {}
            if "encoded" in r:
                known.setdefault(r["index"], set()).update(r["encoded"])
    return known


def run(dataset: str, split: str, shard: int, num_shards: int, max_examples: int | None, output: Path,
        sources: tuple = SOURCES) -> None:
    """Encode and judge the new escalation answers of one shard."""
    answers = {i: {s: r for s, r in a.items() if s in sources} for i, a in escalation_answers(dataset, split).items()}
    answers = {i: a for i, a in answers.items() if a}
    covered = [i for i in v2_indices(dataset, split) if i in answers]
    indices = shard_of(covered, shard, num_shards, max_examples)
    known = known_answers(dataset, split)
    data = cached_split(dataset, split)
    embeddings = question_embeddings(dataset, split)
    heads = {"head": load_head(DEFAULT_HEAD), "probe": load_head(DEFAULT_PROBE)}
    model, processor = load_qwen3_vl(MODEL_SIZE)
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.with_suffix(".jsonl").exists():
        done = {json.loads(l)["index"] for l in output.with_suffix(".jsonl").read_text().splitlines() if l.strip()}
    saved = torch.load(output.with_suffix(".pt"), weights_only=False) if output.with_suffix(".pt").exists() else {"questions": {}}
    print(f"[start] {dataset} {split} shard {shard}/{num_shards}: {len(indices)} questions ({len(covered)} covered), {len(done)} done", flush=True)

    for position, index in enumerate(indices, start=1):
        if index in done:
            continue
        example = data[index]
        try:
            image = extract_image(example, dataset_dir(dataset)).convert("RGB")
            question = extract_question(example)
            prompt = question + PROMPT_SUFFIX
            question_state = embeddings[index].float().unsqueeze(0)
            record = {"dataset": dataset, "split": split, "index": index, "corrections": []}
            seen = known.get(index, set())
            encoded, states, new_answers = {}, [], []
            for source in sources:
                result = answers[index].get(source)
                if result is None:
                    continue
                answer = result["answer"]
                record["corrections"].append({"source": source, "answer": answer, "correct": bool(result["correct"]),
                                              "answer_score": float(result["answer_score"])})
                if answer not in seen and answer not in encoded:
                    state, confidence, tokens = encode_candidate(model, processor, image, prompt, answer)
                    encoded[answer] = {"tf_confidence": confidence, "tokens": tokens, "correct": bool(result["correct"]),
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
        if position % 200 == 0:
            torch.save(saved, output.with_suffix(".pt"))
            print(f"[{dataset} {split} {position}/{len(indices)}] index {index}", flush=True)
    torch.save(saved, output.with_suffix(".pt"))
    print(f"[done] wrote {output.with_suffix('.jsonl')} and {output.with_suffix('.pt')}")


def main() -> None:
    """Parse shard options."""
    parser = argparse.ArgumentParser(description="Encode the escalation answers as candidates for the v3 ranker.")
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=ALL_SOURCES)
    parser.add_argument("--out-dir", default=OUT_DIR.name, help="Folder under results/ (correction_v5 for cot_8b).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if MODEL_SIZE != "2b" and any(s.startswith("up") for s in args.sources):
        raise SystemExit("the enlarged-image answers are 2B answers; with another base model use --sources esc_8b and/or cot_8b")
    output = args.output or RESULTS_ROOT / args.out_dir / args.split / f"{args.dataset}_shard{args.shard}of{args.num_shards}"
    run(args.dataset, args.split, args.shard, args.num_shards, args.max_examples, output, tuple(args.sources))


if __name__ == "__main__":
    main()
