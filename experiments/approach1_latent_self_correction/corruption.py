# corruption.py
#
# Image corruptions for the visual-grounding term of the new loss. A corrupted
# image keeps the question and the model's answer the same but removes the
# visual evidence that supports the answer, so a head that tracks grounding
# should score the pair lower than the clean one. Two modes are supported:
#   swap     replace the image with a different example's image from the same split
#   shuffle  cut the image into a grid of patches and permute them
# No torch or datasets imports, so the logic can be tested on its own.

import random

from PIL import Image


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


def corrupt_image(
    image: Image.Image, mode: str, seed: int, partner: Image.Image | None = None
) -> Image.Image:
    """Return a corrupted copy of an image, using the partner image in swap mode."""
    if mode == "swap":
        if partner is None:
            raise ValueError("swap mode needs a partner image")
        return partner.convert("RGB")
    if mode == "shuffle":
        return shuffle_patches(image, seed)
    raise ValueError(f"unknown corruption mode '{mode}'")
