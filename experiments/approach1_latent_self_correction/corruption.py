# corruption.py
#
# Image corruptions for the visual-grounding term of the new loss. A corrupted
# image keeps the question and the model's answer the same but removes the
# visual evidence that supports the answer, so a head that tracks grounding
# should score the pair lower than the clean one. Four modes are supported:
#   swap     replace the image with a different example's image from the same split
#   shuffle  cut the image into a grid of patches and permute them
#   noise    mix the image with Gaussian noise, x' = sqrt(a) x + sqrt(1 - a) noise
#            (the forward step of a diffusion process), so the scene is still there
#            but its detail is not
#   mask     blank a random share of the image's patches, so part of the evidence is gone
# swap changes what the image shows. The other three keep the same image and degrade it.
# No torch or datasets imports, so the logic can be tested on its own.

import random

import numpy as np
from PIL import Image

DEFAULT_STRENGTH = {"noise": 0.15, "mask": 0.6}


def swap_partners(count: int, seed: int) -> list[int]:
    """Return a permutation of range(count) in which no example is its own partner."""
    if count < 2:
        return list(range(count))
    order = list(range(count))
    random.Random(seed).shuffle(order)
    for position in range(count):
        if order[position] == position:
            other = (position + 1) % count
            order[position], order[other] = order[other], order[position]
    return order


def shuffle_patches(image: Image.Image, seed: int, grid: int = 4) -> Image.Image:
    """Permute a grid x grid set of patches so global statistics survive but content does not."""
    image = image.convert("RGB")
    width, height = image.size
    cell_w, cell_h = max(1, width // grid), max(1, height // grid)
    boxes = [
        (column * cell_w, row * cell_h, (column + 1) * cell_w, (row + 1) * cell_h)
        for row in range(grid)
        for column in range(grid)
    ]
    patches = [image.crop(box) for box in boxes]
    order = list(range(len(patches)))
    random.Random(seed).shuffle(order)
    result = image.copy()
    for box, source in zip(boxes, order):
        result.paste(patches[source], box)
    return result


def add_noise(image: Image.Image, seed: int, alpha: float = DEFAULT_STRENGTH["noise"]) -> Image.Image:
    """Mix an image with Gaussian noise: a small alpha keeps little of the image."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    noise = np.random.default_rng(seed).standard_normal(pixels.shape).astype(np.float32)
    mixed = np.sqrt(alpha) * pixels + np.sqrt(1.0 - alpha) * noise
    return Image.fromarray(((np.clip(mixed, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8))


def mask_patches(image: Image.Image, seed: int, fraction: float = DEFAULT_STRENGTH["mask"], grid: int = 16) -> Image.Image:
    """Blank a random share of the cells of a grid x grid layout with mid-gray."""
    image = image.convert("RGB")
    width, height = image.size
    cells = [(row, column) for row in range(grid) for column in range(grid)]
    random.Random(seed).shuffle(cells)
    result = image.copy()
    for row, column in cells[: int(round(fraction * len(cells)))]:
        box = (column * width // grid, row * height // grid, (column + 1) * width // grid, (row + 1) * height // grid)
        result.paste((128, 128, 128), box)
    return result


def corrupt_image(
    image: Image.Image, mode: str, seed: int, partner: Image.Image | None = None, strength: float | None = None
) -> Image.Image:
    """Return a corrupted copy of an image: the partner in swap mode, or a degraded version of the image."""
    if mode == "swap":
        if partner is None:
            raise ValueError("swap mode needs a partner image")
        return partner.convert("RGB")
    if mode == "shuffle":
        return shuffle_patches(image, seed)
    if mode == "noise":
        return add_noise(image, seed, DEFAULT_STRENGTH["noise"] if strength is None else strength)
    if mode == "mask":
        return mask_patches(image, seed, DEFAULT_STRENGTH["mask"] if strength is None else strength)
    raise ValueError(f"unknown corruption mode '{mode}'")
