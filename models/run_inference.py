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

from models.load_model import load_qwen3_vl


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


def run_inference(size: str, image_path: str, question: str, max_new_tokens: int = 256) -> str:
    """Run a single image and question through the chosen model size.

    Loads the requested model, applies the chat template, generates a
    response, and returns the decoded text. Kept as one function so it
    can be imported and reused by dataset or evaluation scripts, not
    just the CLI below.
    """
    model, processor = load_qwen3_vl(size)
    image = Image.open(image_path).convert("RGB")
    messages = build_messages(image, question)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


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
