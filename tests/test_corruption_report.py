# test_corruption_report.py
#
# Offline test that the corruption report builds from result files with the schema
# written by sweep_mixture.py and sweep_continual.py. The files are synthetic and the
# loader is swapped out, so no cluster output is needed.
#
# Run with `python -m tests.test_corruption_report`, or with pytest.

from experiments.approach1_latent_self_correction import build_corruption_report as report

MODE = report.MODE


def stat(value: float) -> dict:
    """A mean and standard deviation pair as the sweeps store them."""
    return {"mean": value, "std": 0.001}


def grounding_node(value: float) -> dict:
    """A grounding node for correct answers."""
    return {"correct": {"order_accuracy": stat(value), "mean_drop": stat(0.3), "count": stat(100.0)}}


def entry(kind: str, auc: float, ground: float, extras: tuple) -> dict:
    """A winner entry with own and extra-corruption grounding."""
    mode = MODE[kind]
    return {
        "test": {mode: {"correctness_auroc": stat(auc)}},
        "grounding": {mode: {"pooled": grounding_node(ground)}},
        "grounding_extra": {c: {mode: {"pooled": grounding_node(ground - 0.02)}} for c in extras},
    }


def sweep(kind: str, corruption: str) -> dict:
    """A synthetic sweep file for one head kind and training corruption."""
    extras = tuple(c for c in report.CORRUPTIONS if c != corruption)
    return {
        "kind": kind, "train_corruption": corruption, "extra_corruptions": list(extras),
        "rows": {"train": 10, "val": 5, "test": 5}, "datasets": ["a"],
        "winners": {"old": entry(kind, 0.86, 0.90, extras), "new": entry(kind, 0.861, 0.97, extras)},
    }


def continual(kind: str) -> dict:
    """A synthetic sequential file with all four variants."""
    mode = MODE[kind]
    variant = {
        "summary": {mode: {"final_avg": stat(0.85), "forgetting": stat(0.01)}},
        "orders": {"forward": {"grounding_matrix": {mode: [[0.9, 0.9], [0.95, 0.96]]}}},
    }
    return {"variants": {name: variant for name in report.VARIANTS}}


def fake_load(name: str):
    """Serve synthetic files by the names the report asks for."""
    kind = next((k for k in report.KINDS if name.endswith(f"_{k}")), None)
    if kind is None:
        return None
    if "continual" in name:
        return continual(kind)
    corruption = "noise" if ("noise" in name and "xeval" not in name) else "swap"
    return sweep(kind, corruption)


def test_report_builds_and_has_every_section():
    """The report contains each section and no em dashes, and the tables carry the synthetic numbers."""
    original = report.load
    report.load = fake_load
    try:
        html = report.build()
    finally:
        report.load = original
    for anchor in ("summary", "what", "mixture", "transfer", "single", "seq", "limits"):
        assert f'id="{anchor}"' in html, anchor
    assert "—" not in html
    assert "0.970" in html and "0.8600" in html


def test_missing_results_stop_with_a_clear_message():
    """If the pooled sweeps are absent the builder exits instead of writing a partial report."""
    original = report.load
    report.load = lambda name: None
    try:
        try:
            report.build()
        except SystemExit as exc:
            assert "Missing" in str(exc)
            return
    finally:
        report.load = original
    raise AssertionError("expected SystemExit")


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
