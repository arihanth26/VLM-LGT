# build_unseen_splits.py
#
# Builds the two datasets that the correctness heads and the candidate rankers never trained on, used only
# to test whether the correction pipeline transfers to new kinds of images and questions:
#
#   ai2d      AI2D science diagrams, multiple choice (lmms-lab/ai2d, test split, every question).
#             Prompt: the question followed by lettered options, as for ScienceQA; scored by option letter.
#   infovqa   InfographicVQA, free-text answers read off dense infographics (lmms-lab/DocVQA, config
#             InfographicVQA, validation split, every question; its test answers are not public).
#             Scored with ANLS, as DocVQA.
#
# Each is saved in the shared schema of build_qhead_splits.py (image, question, answer, answers, metric,
# group_id, category) as a DatasetDict with a single split named "unseen", under
# data/approach1_latent_self_correction/raw/<name>/. Images are shrunk to the same longest side as the
# in-domain datasets (1536 pixels), so the model sees them under the same conditions.
#
# Usage: python -m data.approach1_latent_self_correction.build_unseen_splits [--dataset ai2d infovqa]

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset

from data.approach1_latent_self_correction.build_qhead_splits import OPTION_LETTERS, RAW_DIR, finalize, resize_image, scienceqa_prompt


def ai2d_prompt(question: str, options: list[str]) -> str:
    """The question with lettered options, in exactly the format of the in-domain ScienceQA prompt."""
    return scienceqa_prompt({"question": question, "choices": options})


def answer_letter(answer, options: list[str]) -> str:
    """The correct option as a letter, whether the source gives an index, a letter or the option text."""
    text = str(answer).strip()
    if text.isdigit():
        return OPTION_LETTERS[int(text)]
    if len(text) == 1 and text.upper() in OPTION_LETTERS:
        return text.upper()
    return OPTION_LETTERS[[o.strip() for o in options].index(text)]


def normalize_ai2d(example: dict, max_side: int) -> dict:
    """One AI2D row in the shared schema."""
    options = [str(o) for o in example["options"]]
    letter = answer_letter(example["answer"], options)
    return {"image": resize_image(example["image"], max_side), "question": ai2d_prompt(example["question"], options),
            "answer": letter, "answers": [letter], "metric": "mcq_letter", "group_id": "", "category": ""}


def normalize_infovqa(example: dict, max_side: int) -> dict:
    """One InfographicVQA row in the shared schema."""
    answers = [str(a) for a in example["answers"]]
    return {"image": resize_image(example["image"], max_side), "question": example["question"], "answer": answers[0],
            "answers": answers, "metric": "anls", "group_id": str(example.get("image_url") or example.get("questionId") or ""),
            "category": ""}


SOURCES = {
    "ai2d": (lambda: load_dataset("lmms-lab/ai2d", split="test"), normalize_ai2d),
    "infovqa": (lambda: load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation"), normalize_infovqa),
}


def main() -> None:
    """Download, normalize and save each requested dataset as a DatasetDict with one 'unseen' split."""
    parser = argparse.ArgumentParser(description="Build the unseen-dataset splits.")
    parser.add_argument("--dataset", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--max-side", type=int, default=1536, help="Longest image side in pixels (as build_qhead_splits).")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    for name in args.dataset:
        target = args.out_dir / name
        if target.is_dir() and any(target.iterdir()):
            print(f"[skip] {name}: already built at {target}")
            continue
        load, normalize = SOURCES[name]
        source = load()
        print(f"[{name}] {len(source)} rows, columns {source.column_names}", flush=True)
        DatasetDict({"unseen": finalize(source, normalize, args.max_side)}).save_to_disk(str(target))
        print(f"[done] {name}: saved to {target}", flush=True)


if __name__ == "__main__":
    main()
