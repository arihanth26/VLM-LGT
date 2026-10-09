# compare_q_heads_report.py
#
# Reads the sweep outputs for each dataset and writes one Markdown report that
# puts five scorers side by side on the test split: geometric-mean token
# confidence, the best separate pre-generation head and verifier head from
# sweep_q_heads.py, and the unified head in its pre-generation and joint
# modes from sweep_unified_q_head.py. Missing files are listed, not fatal, so
# the report can be regenerated as each dataset finishes.

import argparse
import json
from pathlib import Path

import numpy as np

from data.approach1_latent_self_correction.train_q_heads import metrics

HERE = Path(__file__).parent
METRIC_ROWS = (
    ("correctness_auroc", "Correctness AUROC", "higher"),
    ("error_auprc", "Error AUPRC", "higher"),
    ("brier", "Brier score", "lower"),
    ("ece_10_bin", "ECE (10 bins)", "lower"),
    ("pearson_correctness", "Pearson r with correctness", "higher"),
)


def read_json(path: Path) -> dict | None:
    """Load a JSON file, or return None when it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def token_confidence_metrics(baseline_path: Path) -> tuple[dict | None, float | None]:
    """Score raw token confidence against correctness, and report the base model's accuracy."""
    if not baseline_path.exists():
        return None, None
    records = [
        json.loads(line)
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if "prediction" in record]
    labels = np.array([float(record["correct"]) for record in records])
    confidence = np.array([record["prediction"]["confidence"] for record in records])
    return metrics(labels, confidence), float(labels.mean())


def cell(value: dict | None, name: str) -> str:
    """Format one metric cell, with a plus-minus spread when the value came from several seeds."""
    if value is None or name not in value:
        return "n/a"
    item = value[name]
    if isinstance(item, dict):
        return f"{item['mean']:.3f} ± {item['std']:.3f}"
    return f"{item:.3f}"


def dataset_section(name: str, models_dir: Path, results_dir: Path, size: str) -> list[str]:
    """Build the Markdown block for one dataset, or a note saying which files are missing."""
    folder = models_dir / name
    separate = read_json(folder / "sweep_results.json") or read_json(models_dir / "sweep_results.json")
    unified = read_json(folder / "unified_sweep_results.json")
    confidence, accuracy = token_confidence_metrics(results_dir / f"{name}_test_{size}_baseline.jsonl")
    if separate is None and unified is None:
        return [f"## {name}", "", f"No sweep results found under {folder}.", ""]

    columns = {
        "Token confidence": confidence,
        "Separate pre-gen": separate["pre_generation"]["test"] if separate else None,
        "Separate verifier": separate["candidate_verifier"]["test"] if separate else None,
        "Unified, pre-gen mode": unified["test_summary"]["pre_generation"] if unified else None,
        "Unified, joint mode": unified["test_summary"]["joint"] if unified else None,
    }
    lines = [f"## {name}", ""]
    if accuracy is not None:
        lines += [f"Base model accuracy on the test split: {accuracy:.2%}", ""]
    lines += [
        "| Metric | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for key, label, direction in METRIC_ROWS:
        arrow = "↑" if direction == "higher" else "↓"
        lines.append(f"| {label} {arrow} | " + " | ".join(cell(value, key) for value in columns.values()) + " |")
    lines.append("")
    if separate:
        lines.append(
            "Separate heads picked by validation: pre-gen "
            f"{separate['pre_generation']['best_config']}, verifier {separate['candidate_verifier']['best_config']}."
        )
    if unified:
        winner = unified["winner"]
        lines.append(f"Unified winner: {winner['config']}, selected on {unified['select']} validation score.")
    lines += [
        "",
        "Separate heads are a single run each. Unified numbers are mean ± std over 3 seeds.",
        "",
    ]
    return lines


def main() -> None:
    """Parse paths, build a section per dataset, and write the report."""
    parser = argparse.ArgumentParser(description="Compare separate and unified Q-heads.")
    parser.add_argument("--datasets", nargs="+", default=["chartqa_full", "textvqa", "docvqa", "scienceqa_img"])
    parser.add_argument("--models-dir", type=Path, default=HERE / "q_models")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--size", default="2b")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "q_head_comparison.md")
    args = parser.parse_args()

    lines = ["# Q-head comparison", ""]
    for name in args.datasets:
        lines += dataset_section(name, args.models_dir, args.results_dir, args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {args.output}")


if __name__ == "__main__":
    main()
