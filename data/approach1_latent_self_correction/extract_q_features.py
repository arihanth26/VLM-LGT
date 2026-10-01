# extract_q_features.py
#
# Extracts frozen Qwen3-VL representations for pre-generation confidence and
# answer-conditioned verification heads.

import argparse
import json
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from PIL import Image

from data.approach1_latent_self_correction.baseline import extract_image, extract_question
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages

RAW_DIR = Path(__file__).parent / "raw"
FEATURE_DIR = Path(__file__).parent / "q_features"


def load_baseline(path: Path) -> dict[int, dict]:
    """Index completed baseline records by source dataset index."""
    with path.open("r", encoding="utf-8") as source:
        return {
            record["index"]: record
            for line in source
            if line.strip()
            for record in [json.loads(line)]
            if "prediction" in record
        }


def encode_state(
    model, processor, messages: list, add_generation_prompt: bool = False
) -> torch.Tensor:
    """Return the last non-padding token state from a multimodal forward pass."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    last_hidden = outputs.hidden_states[-1]
    position = int(inputs["attention_mask"][0].sum().item()) - 1
    return last_hidden[0, position].detach().float().cpu()


def answer_messages(image: Image.Image, question: str, answer: str) -> list:
    """Build a completed image-question-answer conversation for verifier features."""
    messages = build_messages(image, question)
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]}
    )
    return messages


def extract_features(
    dataset_name: str,
    split: str,
    baseline_path: Path,
    output_path: Path,
    size: str,
    include_references: bool,
) -> None:
    """Extract prompt, generated-answer, and optional reference-answer states."""
    dataset_dir = RAW_DIR / dataset_name
    loaded = load_from_disk(str(dataset_dir))
    if not isinstance(loaded, DatasetDict) or split not in loaded:
        raise ValueError(f"{dataset_name} does not contain split '{split}'")
    split_data = loaded[split]
    baseline = load_baseline(baseline_path)
    if len(baseline) != len(split_data):
        raise ValueError(
            f"baseline has {len(baseline)} records but split has {len(split_data)} examples"
        )

    model, processor = load_qwen3_vl(size)
    prompt_features = []
    generated_features = []
    reference_features = []
    labels = []
    for position, index in enumerate(sorted(baseline), start=1):
        example = split_data[index]
        record = baseline[index]
        image = extract_image(example, dataset_dir)
        question = extract_question(example)
        prompted_question = f"{question}\nRespond with only the final answer, without explanation."
        prompt_features.append(
            encode_state(
                model,
                processor,
                build_messages(image, prompted_question),
                add_generation_prompt=True,
            )
        )
        generated_features.append(
            encode_state(
                model,
                processor,
                answer_messages(image, prompted_question, record["prediction"]["answer"]),
            )
        )
        if include_references:
            reference_features.append(
                encode_state(
                    model,
                    processor,
                    answer_messages(image, prompted_question, record["reference_answer"]),
                )
            )
        labels.append(float(record["correct"]))
        print(f"[{position}/{len(baseline)}] extracted index {index}")

    payload = {
        "dataset": dataset_name,
        "split": split,
        "indices": torch.tensor(sorted(baseline), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "prompt_features": torch.stack(prompt_features).to(torch.float16),
        "generated_features": torch.stack(generated_features).to(torch.float16),
    }
    if include_references:
        payload["reference_features"] = torch.stack(reference_features).to(torch.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[done] wrote {output_path}")


def main() -> None:
    """Parse feature extraction arguments."""
    parser = argparse.ArgumentParser(description="Extract frozen Q-head features.")
    parser.add_argument("--dataset", default="chartqa_full")
    parser.add_argument("--split", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--size", choices=["2b", "4b"], default="2b")
    parser.add_argument("--include-references", action="store_true")
    args = parser.parse_args()
    output = args.output or FEATURE_DIR / f"{args.dataset}_{args.split}_{args.size}.pt"
    extract_features(
        args.dataset,
        args.split,
        args.baseline,
        output,
        args.size,
        args.include_references,
    )


if __name__ == "__main__":
    main()
