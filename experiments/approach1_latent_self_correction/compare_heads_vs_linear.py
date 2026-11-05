# compare_heads_vs_linear.py
#
# Builds a Markdown table that compares the winning trained correctness heads
# with the best linear probe of the same sweep, per dataset. It reads only the
# published sweep files in results/approach1/04_sweep_data, so it runs anywhere.
#
# Two training regimes are compared:
#   mixture   heads trained on the pooled four-dataset mixture, scored per dataset
#   alone     heads trained and tested on one dataset (the per-dataset swap sweeps)
# For each, the "head" is the sweep winner of its loss family (any architecture)
# and the "linear probe" is the best configuration restricted to a linear layer.
# All numbers are test means over three seeds.
#
# If an online correction run has been merged (merge_correction_shards.py), its
# per-dataset detection table for the deployed head and probe is appended.

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO / "results" / "approach1" / "04_sweep_data"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "head_vs_linear_probe.md"
DEFAULT_ONLINE = RESULTS_DIR / "mixture_test_2b_prompted_retry_reranked_full.summary.json"
DATASETS = ("chartqa_full", "textvqa", "docvqa", "scienceqa_img")
LABELS = {"chartqa_full": "ChartQA", "textvqa": "TextVQA", "docvqa": "DocVQA", "scienceqa_img": "ScienceQA (img)"}
ALONE_TAGS = {"chartqa_full": "chartqa", "textvqa": "textvqa", "docvqa": "docvqa", "scienceqa_img": "scienceqa"}
# (kind, mode, column label). The unified head is reported in both of its modes.
KINDS = (
    ("pre", "main", "Pre-generation"),
    ("verifier", "main", "Verifier"),
    ("unified", "joint", "Unified (joint)"),
    ("unified", "pre_generation", "Unified (pre-gen)"),
)
SHORT_ARCH = {"linear": "lin", "mlp_256": "mlp256", "mlp_512": "mlp512"}


def load(path: Path) -> dict:
    """Read one sweep or analysis file."""
    return json.loads(path.read_text(encoding="utf-8"))


def metric(winner: dict, dataset: str, mode: str, name: str, per_dataset: bool) -> tuple[float, float]:
    """Mean and std of one test metric for a winner, pooled or for one dataset."""
    node = winner["test_per_dataset"][dataset] if per_dataset else winner["test"]
    value = node[mode][name]
    return value["mean"], value["std"]


def max_seed_std(name: str) -> float:
    """Largest seed standard deviation of one metric over every entry in the tables."""
    paths = {path for regime in ("mixture", "alone") for dataset in DATASETS for path in sweep_paths(regime, dataset).values()}
    largest = 0.0
    for path in paths:
        winners = load(path)["winners"]
        for group in ("old", "new", "old_linear", "new_linear"):
            for modes in winners[group]["test_per_dataset"].values():
                for values in modes.values():
                    largest = max(largest, values[name]["std"])
    return largest


def cell(head: tuple[float, float], probe: tuple[float, float], arch: str) -> str:
    """Head and probe side by side with the difference, as one table cell group."""
    delta = head[0] - probe[0]
    return f"{head[0]:.3f} ({SHORT_ARCH.get(arch, arch)}) | {probe[0]:.3f} | {delta:+.3f}"


def sweep_paths(regime: str, dataset: str) -> dict[str, Path]:
    """Sweep file of each head kind for one regime and dataset."""
    if regime == "mixture":
        return {kind: SWEEP_DIR / f"mixture_{kind}_heads_sweep.json" for kind in ("pre", "verifier", "unified")}
    tag = ALONE_TAGS[dataset]
    return {kind: SWEEP_DIR / f"corruption_{tag}_swap_{kind}_sweep.json" for kind in ("pre", "verifier", "unified")}


def token_baselines() -> dict[str, dict]:
    """Untrained answer-token confidence per dataset, from the mixture analysis."""
    analysis = load(SWEEP_DIR / "mixture_analysis.json")["baselines"]
    return {name: analysis[name]["token_confidence"] for name in DATASETS}


def regime_table(regime: str, family: str, name: str, tokens: dict) -> list[str]:
    """One Markdown table: rows are datasets, column groups are head kinds."""
    header = ["Dataset", "Token conf."]
    divider = ["---", "---:"]
    for _, _, label in KINDS:
        header += [f"{label} head", f"{label} linear", "Δ"]
        divider += ["---:", "---:", "---:"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(divider) + "|"]
    cache: dict[Path, dict] = {}
    for dataset in DATASETS:
        paths = sweep_paths(regime, dataset)
        row = [LABELS[dataset], f"{tokens[dataset][name]:.3f}"]
        for kind, mode, _ in KINDS:
            path = paths[kind]
            sweep = cache.setdefault(path, load(path))
            head = sweep["winners"][family]
            probe = sweep["winners"][f"{family}_linear"]
            row.append(cell(
                metric(head, dataset, mode, name, True),
                metric(probe, dataset, mode, name, True),
                head["config"]["architecture"],
            ))
        lines.append("| " + " | ".join(row) + " |")
    if regime == "mixture":
        row = ["**Pooled**", f"{load(SWEEP_DIR / 'mixture_analysis.json')['baselines']['pooled']['token_confidence'][name]:.3f}"]
        for kind, mode, _ in KINDS:
            sweep = cache[sweep_paths(regime, DATASETS[0])[kind]]
            head = sweep["winners"][family]
            probe = sweep["winners"][f"{family}_linear"]
            row.append(cell(
                metric(head, "", mode, name, False),
                metric(probe, "", mode, name, False),
                head["config"]["architecture"],
            ))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def online_section(path: Path) -> list[str]:
    """Detection metrics of the deployed head and probe on the full online test run."""
    if not path.exists():
        return []
    detection = load(path).get("detection")
    if not detection:
        return []
    lines = [
        "",
        "## Full online test run: deployed head vs linear probe",
        "",
        f"Source: `{path.relative_to(REPO)}`. Scores are computed on freshly generated greedy answers for every "
        "test example, so these numbers also check that the exported heads reproduce outside the cached features.",
        "",
        "| Dataset | Examples | Accuracy | Token AUROC | Head AUROC | Linear AUROC | Δ head − linear | "
        "Head err. AUPRC | Linear err. AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (*DATASETS, "pooled"):
        row = detection.get(name)
        if not row:
            continue
        label = "**Pooled**" if name == "pooled" else LABELS[name]
        lines.append(
            f"| {label} | {row['examples']} | {row['accuracy']:.1%} | {row['token']['auroc']:.3f} | "
            f"{row['head']['auroc']:.3f} | {row['probe']['auroc']:.3f} | "
            f"{row['head']['auroc'] - row['probe']['auroc']:+.3f} | "
            f"{row['head']['error_auprc']:.3f} | {row['probe']['error_auprc']:.3f} |"
        )
    return lines


def build(online: Path) -> str:
    """Assemble the whole Markdown document."""
    tokens = token_baselines()
    parts = [
        "# Trained correctness heads vs linear probes, per dataset",
        "",
        "Qwen3-VL-2B, frozen. Test set per dataset: ChartQA 2,500, TextVQA 2,500, DocVQA 2,500, ScienceQA 2,017. "
        "Each cell group is *head | linear probe | Δ*, where the head is the sweep winner over all architectures "
        "(architecture in brackets) and the linear probe is the best linear-only configuration of the same sweep, "
        "same loss family, chosen on validation. Means over three seeds; seed std is at most "
        f"{max_seed_std('correctness_auroc'):.3f} for AUROC and {max_seed_std('error_auprc'):.3f} for error AUPRC, "
        "so differences below about 0.01 are within seed noise. "
        "Token conf. is the untrained geometric mean of the answer-token probabilities.",
        "",
        "When the winner is itself linear (ChartQA alone, unified head), Δ is 0 by construction.",
    ]
    for regime, regime_label in (("mixture", "Trained on the pooled mixture"), ("alone", "Trained on each dataset alone")):
        for family, family_label in (("new", "new loss (with grounding term)"), ("old", "old loss")):
            for name, metric_label in (("correctness_auroc", "correctness AUROC"), ("error_auprc", "error AUPRC")):
                parts += ["", f"## {regime_label}, {family_label}: {metric_label}", ""]
                parts += regime_table(regime, family, name, tokens)
    parts += online_section(online)
    return "\n".join(parts) + "\n"


def main() -> None:
    """Write the head-versus-linear-probe tables."""
    parser = argparse.ArgumentParser(description="Compare trained heads with linear probes per dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--online", type=Path, default=DEFAULT_ONLINE, help="Merged prompted-retry summary, if any.")
    args = parser.parse_args()
    args.output.write_text(build(args.online), encoding="utf-8")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
