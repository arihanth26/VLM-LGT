# analyze_candidate_pools.py
#
# Evaluates the candidate-search grid built by candidate_pools.py: every temperature, every pool
# size k (prefixes of the same samples), and every selector, with the measurements in
# selection_metrics.py. The configuration (temperature, k, selector) is chosen on the validation
# pools and the test pools are read once for that choice, so the reported gain is not tuned on
# test. The full test grid is written too, marked as such, for the record.
#
# Writes results/candidate_search.json and results/candidate_search.md.

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.correction_stats import paired_difference
from experiments.approach1_latent_self_correction.selection_metrics import (
    aurc,
    compute_curve,
    evaluate_pool,
    gate_scores,
    outcomes,
)

RESULTS_DIR = Path(__file__).parent / "results"
POOL_DIR = RESULTS_DIR / "pools"
POOL_SIZES = (1, 3, 5, 7, 10)
GATES = ("pre_head", "pre_probe", "head", "probe", "token")
LABELS = {"chartqa_full": "ChartQA", "textvqa": "TextVQA", "docvqa": "DocVQA", "scienceqa_img": "ScienceQA"}


def load_pools(pool_dir: Path) -> dict[float, dict[str, list[dict]]]:
    """Scored pools by temperature and split, one record per question, in dataset and index order."""
    pools: dict[float, dict[str, list[dict]]] = {}
    for folder in sorted(pool_dir.glob("T*")):
        temperature = float(folder.name[1:])
        for split in ("val", "test"):
            records = {}
            for path in sorted((folder / split).glob("*.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        record = json.loads(line)
                        if "greedy" in record:
                            records[(record["dataset"], record["index"])] = record
            if records:
                order = {name: i for i, name in enumerate(DATASET_NAMES)}
                pools.setdefault(temperature, {})[split] = sorted(
                    records.values(), key=lambda r: (order[r["dataset"]], r["index"]))
    return pools


def by_dataset(records: list[dict]) -> dict[str, list[dict]]:
    """Records grouped by dataset, plus the pooled set."""
    groups = {name: [r for r in records if r["dataset"] == name] for name in DATASET_NAMES}
    return {"pooled": records, **{name: rows for name, rows in groups.items() if rows}}


def grid(pools: dict, split: str) -> dict:
    """Pooled and per-dataset evaluation of every temperature and pool size on one split."""
    result = {}
    for temperature, splits in sorted(pools.items()):
        if split not in splits:
            continue
        records = splits[split]
        largest = min(len(r["samples"]) for r in records)
        result[f"{temperature:g}"] = {
            str(k): {name: evaluate_pool(rows, k) for name, rows in by_dataset(records).items()}
            for k in POOL_SIZES if k <= largest
        }
    return result


def choose(validation: dict) -> dict:
    """The (temperature, k, selector) with the best pooled validation accuracy, excluding greedy and oracle."""
    best = None
    for temperature, sizes in validation.items():
        for k, groups in sizes.items():
            for selector, row in groups["pooled"]["selectors"].items():
                if selector in ("greedy", "oracle"):
                    continue
                key = (row["accuracy"], -int(k))  # prefer the smaller pool on a tie
                if best is None or key > best[0]:
                    best = (key, {"temperature": temperature, "k": int(k), "selector": selector,
                                  "val_accuracy": row["accuracy"], "val_gain": row["difference"]})
    return best[1]


def test_report(pools: dict, choice: dict) -> dict:
    """Test numbers for the chosen configuration, pooled and per dataset, plus the gates and compute curves."""
    records = pools[float(choice["temperature"])]["test"]
    k, selector = choice["k"], choice["selector"]
    report = {"choice": choice, "per_dataset": {}, "gates": {}}
    for name, rows in by_dataset(records).items():
        greedy = outcomes(rows, k, "greedy")
        chosen = outcomes(rows, k, selector)
        row = paired_difference(greedy, chosen)
        row.update(greedy_accuracy=float(np.mean(greedy)), accuracy=float(np.mean(chosen)),
                   oracle_accuracy=float(np.mean(outcomes(rows, k, "oracle"))), questions=len(rows))
        report["per_dataset"][name] = row
    greedy = outcomes(records, k, "greedy")
    for gate in GATES:
        scores = gate_scores(records, gate)
        report["gates"][gate] = {**aurc(greedy, scores), "curve": compute_curve(records, scores, k, selector)}
    return report


def markdown(validation: dict, test: dict, report: dict) -> str:
    """Readable tables: the validation grid, the chosen configuration on test, gates, and the test grid."""
    choice = report["choice"]
    lines = ["# Candidate search: temperature, pool size and selector", "",
             "Pool k is the greedy answer plus the first k samples at the given temperature (top-p 0.95). "
             "Mixed pools are questions whose pool holds both a right and a wrong answer; selection can only matter there. "
             "Headroom captured is (selector − greedy) / (oracle − greedy). Grouped AUROC ranks right against wrong "
             "answers within one question. The configuration is chosen on validation; test is read once for it.", "",
             "## Validation grid (pooled)", "",
             "| T | k | Oracle | Mixed pools | Distinct answers | Greedy | Majority | Head best | Head vote | Probe best | Token best | Best anchor | Grouped AUROC head / probe / token |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for temperature, sizes in validation.items():
        for k, groups in sizes.items():
            g = groups["pooled"]
            s, p = g["selectors"], g["pool"]
            anchor = max((v["accuracy"], n) for n, v in s.items() if n.startswith("head_anchor"))
            auc = g["grouped_auroc"]
            lines.append(
                f"| {temperature} | {k} | {p['oracle_accuracy']:.1%} | {p['mixed_fraction']:.1%} | {p['mean_distinct_answers']:.2f} | "
                f"{s['greedy']['accuracy']:.2%} | {s['majority_vote']['accuracy']:.2%} | {s['head_best']['accuracy']:.2%} | "
                f"{s['head_vote']['accuracy']:.2%} | {s['probe_best']['accuracy']:.2%} | {s['token_best']['accuracy']:.2%} | "
                f"{anchor[0]:.2%} ({anchor[1].removeprefix('head_anchor_')}) | "
                f"{auc['head']:.3f} / {auc['probe']:.3f} / {auc['token']:.3f} |")
    lines += ["", f"## Chosen on validation: T = {choice['temperature']}, k = {choice['k']}, selector = `{choice['selector']}`", "",
              f"Validation gain over greedy: {100 * choice['val_gain']:+.2f} points.", "",
              "| Dataset | Questions | Greedy | Chosen | Difference (95% interval) | Fixed / broken | p | Oracle |",
              "|---|---:|---:|---:|---|---|---:|---:|"]
    for name, row in report["per_dataset"].items():
        lines.append(
            f"| {LABELS.get(name, '**Pooled**')} | {row['questions']:,} | {row['greedy_accuracy']:.2%} | {row['accuracy']:.2%} | "
            f"{100 * row['difference']:+.2f} ({100 * row['ci_low']:+.2f} to {100 * row['ci_high']:+.2f}) | "
            f"+{row['fixed']} / -{row['broken']} | {row['p_value']:.3f} | {row['oracle_accuracy']:.2%} |")
    lines += ["", "## Gates on test: which questions should get the extra samples", "",
              "AURC is for the greedy answers (lower is better). The curve sends the lowest-scored fraction of questions "
              "to the chosen selector and keeps greedy elsewhere; gains are in points over greedy.", "",
              "| Gate | AURC | E-AURC | Acc. at 80% coverage | Gain at 10% | 20% | 30% | 50% | 100% | Mean gain |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gate, row in report["gates"].items():
        c = row["curve"]
        lines.append(
            f"| {gate} | {row['aurc']:.4f} | {row['e_aurc']:.4f} | {row['accuracy_at_80']:.2%} | "
            + " | ".join(f"{100 * c[b]['gain']:+.2f}" for b in ("0.1", "0.2", "0.3", "0.5", "1.0"))
            + f" | {100 * c['mean_gain']:+.2f} |")
    lines += ["", "## Test grid (pooled, for the record; not used for the choice)", "",
              "| T | k | Oracle | Greedy | Majority | Head best | Head vote | Mixed-pool acc. head / random | Headroom captured (head best) |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for temperature, sizes in test.items():
        for k, groups in sizes.items():
            s, p = groups["pooled"]["selectors"], groups["pooled"]["pool"]
            head = s["head_best"]
            lines.append(
                f"| {temperature} | {k} | {p['oracle_accuracy']:.1%} | {s['greedy']['accuracy']:.2%} | {s['majority_vote']['accuracy']:.2%} | "
                f"{head['accuracy']:.2%} | {s['head_vote']['accuracy']:.2%} | "
                f"{head.get('mixed_accuracy', float('nan')):.1%} / {head.get('mixed_random_pick', float('nan')):.1%} | "
                f"{head['headroom_captured']:.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Evaluate the grid, choose on validation, report on test."""
    parser = argparse.ArgumentParser(description="Analyze the candidate-search pools.")
    parser.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "candidate_search.json")
    args = parser.parse_args()
    pools = load_pools(args.pool_dir)
    validation, test = grid(pools, "val"), grid(pools, "test")
    report = test_report(pools, choose(validation))
    args.output.write_text(json.dumps({"validation": validation, "test": test, "report": report}, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(validation, test, report), encoding="utf-8")
    print(f"[done] wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
