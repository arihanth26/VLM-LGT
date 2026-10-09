# build_qhead_splits.py
#
# Downloads TextVQA, DocVQA, or ScienceQA and writes each one to raw/<name>
# as a train / val / test DatasetDict in one shared schema, so baseline.py
# and extract_q_features.py run on it unchanged, the same way they do on
# ChartQA.
#
# TextVQA and DocVQA keep their official test answers hidden, so their val
# and test splits here are carved out of the labeled validation split.
# Carving is done by image (TextVQA) or document (DocVQA), so nothing from one
# image or document ever appears in two splits. ScienceQA ships fully labeled splits, so it keeps its
# own, restricted to questions that have an image.
#
# Output columns: image, question, answer (one reference string), answers
# (all references), metric (see scoring.py), group_id, category.

import argparse
import random
from pathlib import Path

from data.approach1_latent_self_correction.dataset_config import QHEAD_DATASETS

RAW_DIR = Path(__file__).parent / "raw"
OPTION_LETTERS = "ABCDEFGH"


def split_by_group(
    groups: list[str], sizes: dict[str, int], seed: int
) -> dict[str, list[int]]:
    """Assign whole groups to named splits in order, so no group spans two splits."""
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    order = sorted(members)
    random.Random(seed).shuffle(order)

    result = {name: [] for name in sizes}
    pending = list(sizes)
    for group in order:
        if not pending:
            break
        name = pending[0]
        result[name].extend(members[group])
        if len(result[name]) >= sizes[name]:
            pending.pop(0)
    return {name: sorted(indices[: sizes[name]]) for name, indices in result.items()}


def resize_image(image, max_side: int):
    """Shrink an image so its longest side is at most max_side, keeping aspect ratio."""
    from PIL import Image

    image = image.convert("RGB")
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def most_common_answer(answers: list[str]) -> str:
    """Return the most frequent non-empty answer, falling back to an empty string."""
    counts: dict[str, int] = {}
    for answer in answers:
        if answer.strip():
            counts[answer] = counts.get(answer, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def scienceqa_prompt(example: dict) -> str:
    """Build a multiple-choice prompt from the hint, question, and lettered options."""
    parts = []
    if example.get("hint"):
        parts.append(f"Context: {example['hint']}")
    parts.append(f"Question: {example['question']}")
    options = "\n".join(
        f"{OPTION_LETTERS[i]}. {choice}" for i, choice in enumerate(example["choices"])
    )
    parts.append(f"Options:\n{options}")
    parts.append("Answer with the letter of the correct option only.")
    return "\n".join(parts)


def normalize_textvqa(example: dict, max_side: int) -> dict:
    """Map one TextVQA row into the shared schema."""
    answers = [str(answer) for answer in example["answers"]]
    return {
        "image": resize_image(example["image"], max_side),
        "question": example["question"],
        "answer": most_common_answer(answers),
        "answers": answers,
        "metric": "vqa_accuracy",
        "group_id": str(example["image_id"]),
        "category": "",
    }


def normalize_docvqa(example: dict, max_side: int) -> dict:
    """Map one DocVQA row into the shared schema."""
    answers = [str(answer) for answer in example["answers"]]
    question_types = example.get("question_types") or []
    return {
        "image": resize_image(example["image"], max_side),
        "question": example["question"],
        "answer": answers[0],
        "answers": answers,
        "metric": "anls",
        "group_id": str(example["ucsf_document_id"]),
        "category": question_types[0] if question_types else "",
    }


def normalize_scienceqa(example: dict, max_side: int) -> dict:
    """Map one ScienceQA row into the shared schema, with the answer as an option letter."""
    letter = OPTION_LETTERS[int(example["answer"])]
    return {
        "image": resize_image(example["image"], max_side),
        "question": scienceqa_prompt(example),
        "answer": letter,
        "answers": [letter],
        "metric": "mcq_letter",
        "group_id": f"{example['subject']}/{example['topic']}",
        "category": str(example["subject"]),
    }


def finalize(subset, normalize_fn, max_side: int):
    """Apply a normalizer to a selected subset and keep only the shared columns."""
    return subset.map(
        lambda example: normalize_fn(example, max_side),
        remove_columns=subset.column_names,
        desc="normalizing",
    )


def take(split, count: int, seed: int):
    """Shuffle a split deterministically and keep at most count rows."""
    return split.shuffle(seed=seed).select(range(min(count, len(split))))


def build_textvqa(entry, args):
    """Train on a sample of the official train split, carve val and test out of validation."""
    from datasets import DatasetDict, load_dataset

    raw = load_dataset(entry.hf_repo_id)
    held_out = split_by_group(
        [str(value) for value in raw["validation"]["image_id"]],
        {"val": args.val_size, "test": args.test_size},
        args.seed,
    )
    return DatasetDict(
        {
            "train": finalize(take(raw["train"], args.train_size, args.seed), normalize_textvqa, args.max_side),
            "val": finalize(raw["validation"].select(held_out["val"]), normalize_textvqa, args.max_side),
            "test": finalize(raw["validation"].select(held_out["test"]), normalize_textvqa, args.max_side),
        }
    )


def build_docvqa(entry, args):
    """Train on a sample of the official train split, carve val and test out of validation."""
    from datasets import DatasetDict, load_dataset

    raw = load_dataset(entry.hf_repo_id)
    held_out = split_by_group(
        [str(value) for value in raw["validation"]["ucsf_document_id"]],
        {"val": args.val_size, "test": args.test_size},
        args.seed,
    )
    return DatasetDict(
        {
            "train": finalize(take(raw["train"], args.train_size, args.seed), normalize_docvqa, args.max_side),
            "val": finalize(raw["validation"].select(held_out["val"]), normalize_docvqa, args.max_side),
            "test": finalize(raw["validation"].select(held_out["test"]), normalize_docvqa, args.max_side),
        }
    )


def build_scienceqa(entry, args):
    """Keep only image questions from each official split, then cap the sizes."""
    from datasets import DatasetDict, load_dataset

    raw = load_dataset(entry.hf_repo_id)
    sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    source_names = {"train": "train", "val": "validation", "test": "test"}
    splits = {}
    for name, source in source_names.items():
        with_image = raw[source].filter(lambda example: example["image"] is not None)
        splits[name] = finalize(take(with_image, sizes[name], args.seed), normalize_scienceqa, args.max_side)
    return DatasetDict(splits)


BUILDERS = {"textvqa": build_textvqa, "docvqa": build_docvqa, "scienceqa_img": build_scienceqa}


def main() -> None:
    """Parse sizes, build the requested datasets, and save them under raw/."""
    parser = argparse.ArgumentParser(description="Build train/val/test splits for the Q-head datasets.")
    parser.add_argument("--dataset", nargs="+", default=["all"], help="textvqa, docvqa, scienceqa_img, or all")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--max-side", type=int, default=1536, help="Longest image side in pixels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    names = list(BUILDERS) if "all" in args.dataset else args.dataset
    for name in names:
        if name not in BUILDERS:
            raise ValueError(f"unknown dataset '{name}'. Available: {', '.join(BUILDERS)}")
        target = args.out_dir / name
        if target.is_dir() and any(target.iterdir()):
            print(f"[skip] {name}: already built at {target}")
            continue
        print(f"[build] {name} from {QHEAD_DATASETS[name].hf_repo_id}")
        built = BUILDERS[name](QHEAD_DATASETS[name], args)
        target.parent.mkdir(parents=True, exist_ok=True)
        built.save_to_disk(str(target))
        print(f"[done] {name}: " + ", ".join(f"{k}={len(v)}" for k, v in built.items()))


if __name__ == "__main__":
    main()
