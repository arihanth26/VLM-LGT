# run_inference.py
#
# Command line entry point for running a Qwen3-VL model on an image and a
# question. This is the quickest way for someone on the project to check
# that a model loads correctly and to compare outputs between the 2B and
# 4B variants on the same input.
#
# Example:
#     python -m models.run_inference --size 4b --image path/to/image.jpg \
#         --question "What is happening in this image?"

import argparse

from PIL import Image

from models.inference_result import InferenceResult
from models.load_model import load_qwen3_vl
from models.uncertainty import summarize_token_log_probabilities


def build_messages(image: Image.Image, question: str) -> list:
    """Build the chat-style message list Qwen3-VL's processor expects.

    Wraps the raw image and question text into the role/content structure
    the chat template needs, so callers only have to pass in plain
    inputs instead of the full message format each time.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]


def generate_with_confidence(model, processor, image: Image.Image, question: str, max_new_tokens: int = 256) -> InferenceResult:
    """Generate one answer and calculate confidence from selected-token probabilities."""
    messages = build_messages(image, question)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, input_length:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        getattr(outputs, "beam_indices", None),
        normalize_logits=True,
    )
    token_log_probabilities = transition_scores[0].detach().float().cpu().tolist()
    confidence, mean_log_probability, minimum_probability = summarize_token_log_probabilities(
        token_log_probabilities
    )
    return InferenceResult(
        answer=response,
        confidence=confidence,
        mean_log_probability=mean_log_probability,
        minimum_token_probability=minimum_probability,
        token_probabilities=[float(value) for value in transition_scores[0].exp().detach().cpu().tolist()],
    )


def run_inference_detailed(
    size: str, image_path: str, question: str, max_new_tokens: int = 256
) -> InferenceResult:
    """Load a model and return an answer together with confidence statistics."""
    model, processor = load_qwen3_vl(size)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    return generate_with_confidence(model, processor, image, question, max_new_tokens)


def run_inference(size: str, image_path: str, question: str, max_new_tokens: int = 256) -> str:
    """Run a single image and question through the chosen model size.

    Loads the requested model, applies the chat template, generates a
    response, and returns the decoded text. Kept as one function so it
    can be imported and reused by dataset or evaluation scripts, not
    just the CLI below.
    """
    return run_inference_detailed(size, image_path, question, max_new_tokens).answer


def main() -> None:
    """Parse command line arguments and print the model's response."""
    parser = argparse.ArgumentParser(description="Run Qwen3-VL on an image and a question.")
    parser.add_argument("--size", choices=["2b", "4b"], default="4b", help="Model size to load.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--question", required=True, help="Question to ask about the image.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length cap.")
    args = parser.parse_args()

    response = run_inference(args.size, args.image, args.question, args.max_new_tokens)
    print(response)


if __name__ == "__main__":
    main()
