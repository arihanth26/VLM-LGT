# model_config.py
#
# Central place that lists the Qwen3-VL model variants used in this project
# and the settings needed to load them. Anyone adding a new model size or
# swapping in a fine-tuned checkpoint should add it here instead of
# hardcoding a repo id inside another script.

from dataclasses import dataclass


@dataclass
class ModelVariant:
    """Holds the info needed to load one Qwen3-VL checkpoint."""

    name: str
    hf_repo_id: str
    description: str


# Registry of the model variants available for experimentation. "2b" and
# "4b" refer to parameter count and are the two sizes the project has
# standardized on, so results stay comparable across experiments.
MODEL_VARIANTS = {
    "2b": ModelVariant(
        name="qwen3-vl-2b",
        hf_repo_id="Qwen/Qwen3-VL-2B-Instruct",
        description="Smaller Qwen3-VL checkpoint, good for fast local iteration.",
    ),
    "4b": ModelVariant(
        name="qwen3-vl-4b",
        hf_repo_id="Qwen/Qwen3-VL-4B-Instruct",
        description="Larger Qwen3-VL checkpoint, closer to what full experiments will use.",
    ),
}


def get_variant(size: str) -> ModelVariant:
    """Look up a model variant by its short size key ('2b' or '4b').

    Raises a clear error if the given size is not registered, instead of
    letting model loading fail later with a less obvious message.
    """
    size = size.lower()
    if size not in MODEL_VARIANTS:
        available = ", ".join(MODEL_VARIANTS.keys())
        raise ValueError(f"Unknown model size '{size}'. Available sizes: {available}")
    return MODEL_VARIANTS[size]
