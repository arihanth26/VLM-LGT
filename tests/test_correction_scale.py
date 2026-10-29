# test_correction_scale.py
#
# Offline tests for the helpers that run the correction experiments on full test splits:
# the shard ranges must split a dataset exactly once, and merging shard files must give one
# record per example in dataset order. They need no GPU, no network, and no model packages.
#
# Run with `python -m tests.test_correction_scale`, or with pytest.

import json
import tempfile
from pathlib import Path

from experiments.approach1_latent_self_correction import merge_correction_shards as merge
from experiments.approach1_latent_self_correction.correction_scale import shard_range


def test_shards_cover_each_example_exactly_once():
    """For any size and shard count the blocks are disjoint and together equal the whole split."""
    for count in (0, 1, 7, 2017, 2500):
        for num_shards in (1, 3, 4, 5):
            seen = []
            for shard in range(num_shards):
                seen.extend(shard_range(count, shard, num_shards))
            assert seen == list(range(count)), (count, num_shards)


def test_shard_cap_and_bounds():
    """A smoke-test cap limits one shard, and a shard outside the range is an error."""
    assert len(shard_range(2500, 1, 4, max_examples=8)) == 8
    assert shard_range(2500, 1, 4, max_examples=8).start == 625
    try:
        shard_range(10, 4, 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_merge_reads_shards_in_dataset_order_and_keeps_the_last_duplicate():
    """Records from several shard files come back ordered by dataset then index, once each."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        rows_a = [{"dataset": "textvqa", "index": 5, "v": "old"}, {"dataset": "chartqa_full", "index": 2, "v": "a"}]
        rows_b = [{"dataset": "textvqa", "index": 5, "v": "new"}, {"dataset": "chartqa_full", "index": 1, "v": "b"}]
        (root / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows_a), encoding="utf-8")
        (root / "b.jsonl").write_text("\n".join(json.dumps(r) for r in rows_b) + "\n", encoding="utf-8")
        merged = merge.read_shards(root)
    assert [(r["dataset"], r["index"]) for r in merged] == [("chartqa_full", 1), ("chartqa_full", 2), ("textvqa", 5)]
    assert merged[-1]["v"] == "new"


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
