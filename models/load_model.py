# load_model.py
#
# Loads a Qwen3-VL model and its processor so other scripts (inference,
# dataset tooling, training, evaluation) can share one consistent loading
# path. Supports both the 2B and 4B variants defined in model_config.py.
#
# Usage:
#     from models.load_model import load_qwen3_vl
#     model, processor = load_qwen3_vl("4b")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from models.model_config import MODEL_VARIANTS, get_variant


def load_qwen3_vl(size: str = "4b", device: str = None, dtype=torch.bfloat16):
    """Load a Qwen3-VL model and its processor for the given size.

    Args:
        size: "2b" or "4b", matching the keys in MODEL_VARIANTS.
        device: torch device string. If None, picks cuda when available
            and falls back to cpu otherwise.
        dtype: torch dtype to load weights in. bfloat16 by default to
            keep memory usage reasonable on a single GPU.

    Returns:
        A (model, processor) tuple ready for inference.
    """
    variant = get_variant(size)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(variant.hf_repo_id)
    model = AutoModelForImageTextToText.from_pretrained(
        variant.hf_repo_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    return model, processor


def list_available_models() -> None:
    """Print every registered model size and its Hugging Face repo id.

    Meant as a quick reference for anyone new to the project who wants
    to see what is available without opening model_config.py directly.
    """
    for key, variant in MODEL_VARIANTS.items():
        print(f"{key}: {variant.hf_repo_id} - {variant.description}")


if __name__ == "__main__":
    list_available_models()
