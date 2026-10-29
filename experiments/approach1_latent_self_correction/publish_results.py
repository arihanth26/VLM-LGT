# publish_results.py
#
# Copies the Approach 1 results from their working locations (inside this
# folder's results/ and q_models/) into the numbered, top-level
# results/approach1/ folder, so the progression of the work reads in order.
# The working locations are left as they are. Rerun it after any experiment
# to refresh the published copies; datasets whose files are missing are skipped.
#
#   01  original ChartQA study report
#   02  multi-dataset report with charts
#   03  side-by-side metric tables
#   04  machine-readable sweep and analysis files, per dataset
#   05  dataset-mixture and new-loss report
#   06  corrupted-image (noise) version of the new loss
#   07  correction experiments (prompted retry, reranking, best-of-five, deeper pass)
#   08  complete Approach 1 report (HTML and PDF) covering everything above, and its Excel workbook of tables

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]
OUTPUT = REPO / "results" / "approach1"
DATASETS = ("chartqa_full", "textvqa", "docvqa", "scienceqa_img")

NUMBERED_FILES = [
    (HERE / "results" / "confidence_comparison_report.html", "01_chartqa_original_study_report.html"),
    (HERE / "results" / "q_head_multidataset_report.html", "02_multidataset_qhead_report.html"),
    (HERE / "results" / "q_head_comparison.md", "03_qhead_comparison_tables.md"),
    (HERE / "results" / "q_head_mixture_report.html", "05_mixture_new_loss_report.html"),
    (HERE / "results" / "q_head_corruption_report.html", "06_corruption_new_loss_report.html"),
    (HERE / "results" / "correction_experiments_report.html", "07_correction_experiments_report.html"),
    (HERE / "results" / "approach1_complete_report.html", "08_approach1_complete_report.html"),
    (HERE / "results" / "approach1_complete_report.pdf", "08_approach1_complete_report.pdf"),
    (HERE / "results" / "approach1_results.xlsx", "08_approach1_results.xlsx"),
]


def copy_if_present(source: Path, target: Path) -> bool:
    """Copy one file, creating parent folders, and report whether the source existed."""
    if not source.exists():
        print(f"[skip] {source.name}: not found")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"[copy] {target.relative_to(REPO)}")
    return True


def sweep_files(dataset: str) -> list[tuple[Path, str]]:
    """List the sweep and analysis files for one dataset with their published names."""
    folder = HERE / "q_models" / dataset
    return [
        (folder / "sweep_results.json", f"{dataset}_separate_heads_sweep.json"),
        (folder / "unified_sweep_results.json", f"{dataset}_unified_head_sweep.json"),
        (HERE / "results" / f"analysis_{dataset}.json", f"{dataset}_analysis.json"),
    ]


def main() -> None:
    """Parse arguments and refresh every published file."""
    parser = argparse.ArgumentParser(description="Publish Approach 1 results to results/approach1.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    for source, name in NUMBERED_FILES:
        copy_if_present(source, args.output / name)
    copy_if_present(
        HERE / "results" / "chartqa_original_validation_runs.json",
        args.output / "04_sweep_data" / "chartqa_original_validation_runs.json",
    )
    for dataset in DATASETS:
        for source, name in sweep_files(dataset):
            copy_if_present(source, args.output / "04_sweep_data" / name)
    for kind in ("pre", "verifier", "unified"):
        copy_if_present(
            HERE / "q_models" / "mixture" / f"mixture_{kind}_results.json",
            args.output / "04_sweep_data" / f"mixture_{kind}_heads_sweep.json",
        )
        copy_if_present(
            HERE / "q_models" / "mixture" / f"continual_{kind}_results.json",
            args.output / "04_sweep_data" / f"mixture_{kind}_sequential.json",
        )
    copy_if_present(HERE / "results" / "analysis_mixture.json", args.output / "04_sweep_data" / "mixture_analysis.json")
    publish_corruption_data(args.output)
    publish_correction_data(args.output)


def publish_correction_data(output: Path) -> None:
    """Copy the summaries and per-example records of the correction experiments."""
    for path in sorted((HERE / "results").glob("mixture_test_2b_*_limit100.*")):
        copy_if_present(path, output / "04_sweep_data" / f"correction_{path.name}")


def publish_corruption_data(output: Path) -> None:
    """Copy the sweep, sequential, and analysis files of the corrupted-image experiments."""
    folder = HERE / "q_models" / "mixture"
    data = output / "04_sweep_data"
    names = ["mixture_noise", "mixture_xeval"] + [f"{d}_{c}" for d in ("chartqa", "textvqa", "docvqa", "scienceqa") for c in ("swap", "noise")]
    for kind in ("pre", "verifier", "unified"):
        for name in names:
            copy_if_present(folder / f"{name}_{kind}_results.json", data / f"corruption_{name}_{kind}_sweep.json")
        copy_if_present(folder / f"mixture_noise_continual_{kind}_results.json", data / f"corruption_mixture_noise_{kind}_sequential.json")
    copy_if_present(HERE / "results" / "analysis_mixture_noise.json", data / "corruption_mixture_noise_analysis.json")


if __name__ == "__main__":
    main()
