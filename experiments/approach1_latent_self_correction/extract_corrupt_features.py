# extract_corrupt_features.py
#
# Caches Qwen3-VL states for the same question and the same generated answer
# when the image is replaced by a corrupted one (see corruption.py). These are
# the h_corrupt inputs for the visual-grounding term of the new loss. The same
# pass also records the model's own answer confidence, teacher-forced on the
# fixed answer text, for the clean and the corrupted image, so token confidence
# can be tested on grounding with identical conditions on both sides.
#
# Output per dataset split: <dataset>_<split>_<size>_corrupt.pt with
#   prompt_features_corrupt, generated_features_corrupt   fp16 states
#   confidence_clean, confidence_corrupt                  teacher-forced geometric-mean answer probability

import argparse
import math
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk

from experiments.approach1_latent_self_correction.baseline import extract_image, extract_question
from experiments.approach1_latent_self_correction.corruption import corrupt_image, swap_partners
from experiments.approach1_latent_self_correction.extract_q_features import (
    FEATURE_DIR,
    RAW_DIR,
    answer_messages,
    encode_state,
    load_baseline,
)
from models.load_model import load_qwen3_vl
from models.run_inference import build_messages

ANSWER_SUFFIX = "\nRespond with only the final answer, without explanation."
TRAILING_TOKENS = 2  # the end-of-turn token and the newline that close the assistant message


def encode_with_confidence(model, processor, messages: list, answer: str) -> tuple[torch.Tensor, float]:
    """Return the last-token state and the geometric-mean probability of the answer tokens."""
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    ).to(model.device)
    answer_ids = processor.tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    count = len(answer_ids)
    length = int(inputs["input_ids"].shape[1])
    keep = count + 3
    with torch.inference_mode():
        try:
            outputs = model(
                **inputs, output_hidden_states=True, use_cache=False, return_dict=True, logits_to_keep=keep
            )
        except TypeError:
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    state = outputs.hidden_states[-1][0, -1].detach().float().cpu()
    if count == 0:
        return state, float("nan")
    start = length - TRAILING_TOKENS - count
    found = inputs["input_ids"][0, start : start + count].tolist()
    if found != list(answer_ids):
        return state, float("nan")
    logits = outputs.logits[0]
    logits = logits[length - keep :] if logits.shape[0] == length else logits
    log_probs = torch.log_softmax(logits[:count].float(), dim=-1)
    chosen = log_probs.gather(1, torch.tensor(answer_ids, device=log_probs.device)[:, None]).squeeze(1)
    return state, float(torch.exp(chosen.mean()).item())


def extract_corrupt(
    dataset_name: str, split: str, baseline_path: Path, output_path: Path, size: str,
    mode: str, seed: int, limit: int | None, strength: float | None = None,
) -> None:
    """Run the corrupted-image passes for every example of one split and save the states."""
    dataset_dir = RAW_DIR / dataset_name
    loaded = load_from_disk(str(dataset_dir))
    if not isinstance(loaded, DatasetDict) or split not in loaded:
        raise ValueError(f"{dataset_name} does not contain split '{split}'")
    split_data = loaded[split]
    baseline = load_baseline(baseline_path)
    indices = sorted(baseline)
    if limit is not None:
        indices = indices[:limit]
    partners = swap_partners(len(split_data), seed)

    model, processor = load_qwen3_vl(size)
    prompt_corrupt, generated_corrupt, conf_clean, conf_corrupt = [], [], [], []
    for position, index in enumerate(indices, start=1):
        example = split_data[index]
        record = baseline[index]
        answer = record["prediction"]["answer"]
        prompted = f"{extract_question(example)}{ANSWER_SUFFIX}"
        image = extract_image(example, dataset_dir)
        partner = extract_image(split_data[partners[index]], dataset_dir) if mode == "swap" else None
        bad = corrupt_image(image, mode, seed + index, partner, strength)

        _, clean_conf = encode_with_confidence(model, processor, answer_messages(image, prompted, answer), answer)
        prompt_corrupt.append(
            encode_state(model, processor, build_messages(bad, prompted), add_generation_prompt=True)
        )
        state, corrupt_conf = encode_with_confidence(
            model, processor, answer_messages(bad, prompted, answer), answer
        )
        generated_corrupt.append(state)
        conf_clean.append(clean_conf)
        conf_corrupt.append(corrupt_conf)
        if position % 50 == 0 or position == len(indices):
            print(f"[{position}/{len(indices)}] corrupt features for index {index}", flush=True)

    payload = {
        "dataset": dataset_name,
        "split": split,
        "mode": mode,
        "strength": -1.0 if strength is None else float(strength),
        "indices": torch.tensor(indices, dtype=torch.long),
        "prompt_features_corrupt": torch.stack(prompt_corrupt).to(torch.float16),
        "generated_features_corrupt": torch.stack(generated_corrupt).to(torch.float16),
        "confidence_clean": torch.tensor(conf_clean, dtype=torch.float32),
        "confidence_corrupt": torch.tensor(conf_corrupt, dtype=torch.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    missing = sum(math.isnan(v) for v in conf_clean)
    generation = [baseline[i]["prediction"]["confidence"] for i in indices]
    pairs = [(a, b) for a, b in zip(conf_clean, generation) if not math.isnan(a)]
    if len(pairs) > 2:
        first = torch.tensor([a for a, _ in pairs])
        second = torch.tensor([b for _, b in pairs])
        correlation = float(torch.corrcoef(torch.stack([first, second]))[0, 1])
        print(f"[check] teacher-forced vs generation confidence: r={correlation:.4f}, {missing} answers without a match")
    print(f"[done] wrote {output_path}")


def main() -> None:
    """Parse arguments and extract the corrupted-image features."""
    parser = argparse.ArgumentParser(description="Extract corrupted-image Q-head features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--size", choices=["2b", "4b"], default="2b")
    parser.add_argument("--mode", choices=["swap", "shuffle", "noise", "mask"], default="swap")
    parser.add_argument("--strength", type=float, default=None, help="Noise alpha or masked share, default per mode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N examples, for a smoke test.")
    args = parser.parse_args()
    suffix = "" if args.mode == "swap" else f"_{args.mode}"
    output = args.output or FEATURE_DIR / f"{args.dataset}_{args.split}_{args.size}_corrupt{suffix}.pt"
    extract_corrupt(args.dataset, args.split, args.baseline, output, args.size, args.mode, args.seed, args.limit, args.strength)


if __name__ == "__main__":
    main()
