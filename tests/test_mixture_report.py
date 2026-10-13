# test_mixture_report.py
#
# Offline test that the mixture report builder works on the real output schema of
# sweep_mixture.py. It runs a tiny quick sweep for each head kind on synthetic
# features, writes the result files into a temporary folder together with a small
# synthetic analysis file, and checks that the HTML report is produced with its
# sections and without leftover placeholders. No GPU or network.
#
# Run with `python -m tests.test_mixture_report`, or with pytest.

import base64
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.approach1_latent_self_correction import build_mixture_report as report
from experiments.approach1_latent_self_correction import continual, sweep_continual
from experiments.approach1_latent_self_correction import sweep_mixture
from experiments.approach1_latent_self_correction.analyze_q_heads import GRID, accuracy_at_coverage, error_capture
from tests.test_mixture import make_split, tensors_of

DEVICE = torch.device("cpu")
NAMES = ["chartqa_full", "textvqa"]


def write_fixture(folder: Path) -> None:
    """Write synthetic sweep results and a synthetic analysis file in the real schema."""
    train, validation, test = (tensors_of(make_split(n, s)) for n, s in ((300, 1), (120, 2), (120, 3)))
    labels = test["labels"].numpy()
    ids = test["dataset_id"].numpy()
    subsets = {"pooled": np.ones(len(labels), bool), **{name: ids == i for i, name in enumerate(NAMES)}}
    curves = {}
    for kind in ("pre", "verifier", "unified"):
        result, scores = sweep_mixture.run_sweep(kind, train, validation, test, NAMES, DEVICE, True, 2, 2, 128)
        result["datasets"] = NAMES
        result["rows"] = {"train": 300, "val": 120, "test": 120}
        (folder / f"mixture_{kind}_results.json").write_text(json.dumps(result), encoding="utf-8")
        per_train = continual.split_by_dataset(train, 2)
        per_val = continual.split_by_dataset(validation, 2)
        per_test = continual.split_by_dataset(test, 2)
        variants = {}
        for loss in ("old", "new"):
            config = dict(result["winners"][loss]["config"])
            for replay in (0.0, 0.1):
                variants[f"{loss}_replay{replay:g}"] = sweep_continual.run_variant(
                    kind, config, {"forward": NAMES, "reverse": NAMES[::-1]}, per_train, per_val, per_test, NAMES,
                    replay, DEVICE, 2, 2, 128,
                )
        (folder / f"continual_{kind}_results.json").write_text(
            json.dumps({"kind": kind, "datasets": NAMES, "seeds": [1, 2, 3], "variants": variants}), encoding="utf-8"
        )
        curves[kind] = {}
        for winner in ("old", "new"):
            curves[kind][winner] = {}
            for mode, sides in scores[winner].items():
                clean = sides["clean"].float().numpy()
                curves[kind][winner][mode] = {
                    key: {
                        "accuracy_at_coverage": np.mean([accuracy_at_coverage(labels[m], run[m]) for run in clean], axis=0).tolist(),
                        "error_capture": np.mean([error_capture(labels[m], run[m]) for run in clean], axis=0).tolist(),
                    }
                    for key, m in subsets.items()
                }
    token = np.random.default_rng(0).random(len(labels))
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), "gray").save(buffer, format="JPEG")
    tiny = base64.b64encode(buffer.getvalue()).decode()
    baseline = {
        "token_confidence": {"correctness_auroc": 0.7, "error_auprc": 0.4, "brier": 0.15, "ece_10_bin": 0.1, "pearson_correctness": 0.3},
        "exactly_one": 0.8,
        "answer_length": {"auroc": 0.5, "error_auprc": 0.2},
        "grounding_token_confidence": {"correct": {"mean_drop": 0.2, "order_accuracy": 0.7, "count": 50}, "all": {"mean_drop": 0.2, "order_accuracy": 0.7, "count": 60}},
    }
    analysis = {
        "datasets": NAMES,
        "grid": GRID,
        "composition": {s: {"rows": n, "accuracy": 0.8, "per_dataset": {d: {"rows": n // 2, "accuracy": 0.8} for d in NAMES}} for s, n in (("train", 300), ("val", 120), ("test", 120))},
        "baselines": {key: baseline for key in subsets},
        "token_curves": {key: {"accuracy_at_coverage": accuracy_at_coverage(labels[m], token[m]), "error_capture": error_capture(labels[m], token[m])} for key, m in subsets.items()},
        "curves": curves,
        "gallery": [{"dataset": d, "clean": tiny, "corrupt": tiny, "question": "q"} for d in NAMES],
    }
    (folder / "analysis_mixture.json").write_text(json.dumps(analysis), encoding="utf-8")


def test_report_builds_from_the_real_schema():
    """The report is produced from sweep outputs and contains every section."""
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        write_fixture(folder)
        saved = (report.MIXTURE_DIR, report.ANALYSIS, report.OUTPUT)
        report.MIXTURE_DIR, report.ANALYSIS, report.OUTPUT = folder, folder / "analysis_mixture.json", folder / "report.html"
        try:
            report.main()
            html = report.OUTPUT.read_text(encoding="utf-8")
        finally:
            report.MIXTURE_DIR, report.ANALYSIS, report.OUTPUT = saved
    for anchor in ("summary", "setup", "exp1", "exp2", "exp3", "seq", "grounding", "ablation", "generalist", "use", "meaning", "limits"):
        assert f'id="{anchor}"' in html, anchor
    assert "—" not in html
    assert html.count("<svg") > 10


def _run_all() -> None:
    """Run every test_* function in this module and report pass and fail counts."""
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
