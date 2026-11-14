# Tests for the escalation study: image enlargement limits and the routing simulation.

import numpy as np
from PIL import Image

from experiments.approach1_latent_self_correction.analyze_escalation import cost, curves, final, headline, routed
from experiments.approach1_latent_self_correction.escalation import upscale, upscaled_size


def test_upscaled_size_respects_scale_and_cap():
    assert upscaled_size(1000, 800, 2.0, max_pixels=8_000_000) == (2000, 1600, 2.0)
    width, height, scale = upscaled_size(2000, 2000, 2.0, max_pixels=8_000_000)
    assert width * height <= 8_000_000 * 1.001 and 1.0 < scale < 2.0
    assert upscaled_size(4000, 3000, 2.0, max_pixels=8_000_000)[2] == 1.0  # already above the cap: never shrink


def test_upscale_returns_enlarged_image():
    image, scale = upscale(Image.new("RGB", (100, 50)), 1.5)
    assert image.size == (150, 75) and scale == 1.5


def test_routed_uses_reference_threshold():
    val = {"token": np.array([0.9, 0.1, 0.5, 0.3])}
    assert routed({"greedy": np.zeros(4), "token": val["token"]}, "token", 0.5, val).tolist() == [False, True, False, True]
    # On another split the same threshold applies question by question.
    assert routed({"greedy": np.zeros(3), "token": np.array([0.2, 0.25, 0.95])}, "token", 0.5, val).tolist() == [True, True, False]


def test_rules():
    arrays = {"greedy": np.array([True, False, False]), "fix": np.array([False, True, True]),
              "token": np.array([0.9, 0.2, 0.8]), "fix_token": np.array([0.5, 0.6, 0.4])}
    mask = np.array([True, True, True])
    assert final(arrays, mask, "replace").tolist() == [False, True, True]
    assert final(arrays, mask, "confident").tolist() == [True, True, False]


def test_headline_chooses_on_validation():
    rng = np.random.default_rng(0)
    n = 400
    def arrays():
        greedy = rng.random(n) < 0.7
        a = {"dataset": np.array(["textvqa"] * n), "greedy": greedy, "fix": greedy | (rng.random(n) < 0.5),
             "token": rng.random(n), "fix_token": rng.random(n)}
        for d in ("pre_head", "pre_probe", "after_probe", "token"):
            a[d] = rng.random(n)
        a["after_head"] = greedy + 0.1 * rng.random(n)  # a perfect detector
        return a
    val, test = arrays(), arrays()
    h = headline(val, test, curves(val, val), "x", 0.3)
    assert h["detector"] == "after_head" and h["test"]["pooled"]["difference"] > 0


def test_cost_skips_2b_for_pre_generation_replace():
    assert cost("pre_head", "replace", 0.2, "8b") == 0.8 + 0.8
    assert cost("after_head", "replace", 0.2, "8b") == 1.0 + 0.8
