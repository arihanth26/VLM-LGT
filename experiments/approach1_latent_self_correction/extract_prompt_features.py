# extract_prompt_features.py
#
# The constant question embedding for a dataset split, without a baseline run: for every question, the
# last-token hidden state of the frozen 2B model on the image and the prompted question with the generation
# prompt, exactly as extract_q_features.py computes "prompt_features". Used for the unseen datasets
# (build_unseen_splits.py), where the heads only need the question embedding: the pools score every answer
# against it (candidate_pools.py), and the pre-generation head reads it alone.
#
# Output: q_features/<dataset>_<split>_2b.pt with "indices" and "prompt_features" (float16), the two keys
# candidate_pools.question_embeddings reads. --check compares against an existing extract_q_features.py file
# on its first questions, to confirm the two produce the same embedding.

import argparse
from pathlib import Path

import torch

from experiments.approach1_latent_self_correction.baseline import extract_image, extract_question
from experiments.approach1_latent_self_correction.candidate_pools import FEATURE_DIR, PROMPT_SUFFIX, cached_split
from experiments.approach1_latent_self_correction.correction_scale import ALL_DATASETS, SPLITS, dataset_dir
from experiments.approach1_latent_self_correction.extract_q_features import encode_state
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages


def prompt_state(model, processor, example: dict, dataset: str) -> torch.Tensor:
    """The question embedding of one example (same messages and suffix as extract_q_features.py)."""
    image = extract_image(example, dataset_dir(dataset))
    return encode_state(model, processor, build_messages(image, extract_question(example) + PROMPT_SUFFIX), add_generation_prompt=True)


def run(dataset: str, split: str, output: Path) -> None:
    """Embed every question of the split and save them in the extract_q_features.py format."""
    data = cached_split(dataset, split)
    model, processor = load_qwen3_vl("2b")
    states = []
    for index in range(len(data)):
        states.append(prompt_state(model, processor, data[index], dataset))
        if (index + 1) % 250 == 0:
            print(f"[{dataset} {split}] {index + 1}/{len(data)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": dataset, "split": split, "indices": torch.arange(len(data)),
                "prompt_features": torch.stack(states).to(torch.float16)}, output)
    print(f"[done] wrote {output}")


def check(dataset: str, split: str, count: int = 5) -> None:
    """Cosine similarity with the embeddings extract_q_features.py saved, on the first `count` questions."""
    cached = torch.load(FEATURE_DIR / f"{dataset}_{split}_2b.pt", map_location="cpu", weights_only=True)
    data = cached_split(dataset, split)
    model, processor = load_qwen3_vl("2b")
    for row, index in enumerate(cached["indices"].tolist()[:count]):
        fresh = prompt_state(model, processor, data[index], dataset)
        saved = cached["prompt_features"][row].float()
        cosine = torch.nn.functional.cosine_similarity(fresh, saved, dim=0).item()
        print(f"[check] {dataset} {split} {index}: cosine {cosine:.6f}", flush=True)


def main() -> None:
    """Parse options."""
    parser = argparse.ArgumentParser(description="Question embeddings for a dataset split.")
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--check", action="store_true", help="Compare with the existing extract_q_features.py file instead.")
    args = parser.parse_args()
    if args.check:
        check(args.dataset, args.split)
        return
    run(args.dataset, args.split, FEATURE_DIR / f"{args.dataset}_{args.split}_2b.pt")


if __name__ == "__main__":
    main()
