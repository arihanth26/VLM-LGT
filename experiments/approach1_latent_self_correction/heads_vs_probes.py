# heads_vs_probes.py
#
# Compares the trained heads with linear probes, dataset by dataset. A linear probe is the best
# linear head of a sweep (no hidden layer, the strongest evidence that the signal is already
# linearly readable); a trained head is the best head of the same sweep over all architectures
# (linear, 256-unit MLP, 512-unit MLP), both chosen on validation with the old loss. Two settings
# are compared: heads trained on the pooled mixture and scored on each dataset's rows, and heads
# trained on each dataset alone. For every head kind and mode it reports AUROC, error AUPRC,
# Brier score and ECE (mean and standard deviation over three seeds), the architecture and
# parameter count of each, and the difference. The untrained answer-token probability is given as
# the reference. Used by the report, the Excel workbook, and the markdown table in results/.
# Usage: python -m experiments.approach1_latent_self_correction.heads_vs_probes

import json
from pathlib import Path

from experiments.approach1_latent_self_correction import build_corruption_report as corr
from experiments.approach1_latent_self_correction.build_approach1_report import params

HERE = Path(__file__).parent
OUTPUT = HERE / "results" / "heads_vs_linear_probes.md"
SWEEP_DIR = HERE.parents[1] / "results" / "approach1" / "04_sweep_data"
DATASETS = [("chartqa", "chartqa_full", "ChartQA"), ("textvqa", "textvqa", "TextVQA"), ("docvqa", "docvqa", "DocVQA"), ("scienceqa", "scienceqa_img", "ScienceQA")]
KINDS = {"pre": ("main",), "verifier": ("main",), "unified": ("pre_generation", "joint")}
LABEL = {("pre", "main"): "Pre-generation head", ("verifier", "main"): "Verifier head", ("unified", "pre_generation"): "Unified, pre-generation mode", ("unified", "joint"): "Unified, joint mode"}
METRICS = ("correctness_auroc", "error_auprc", "brier", "ece_10_bin")


def token_auroc(key: str) -> float:
    """AUROC of the untrained answer-token probability on one dataset's test split."""
    analysis = json.loads((SWEEP_DIR / f"{key}_analysis.json").read_text(encoding="utf-8"))
    return analysis["token_confidence_metrics"]["correctness_auroc"]


def pick(node: dict, metric: str):
    """(mean, std) of one metric in a sweep winner's test node."""
    return node[metric]["mean"], node[metric]["std"]


def comparison_rows() -> list[dict]:
    """One row per setting, dataset, head kind and mode with linear probe, trained head, and their difference."""
    rows = []
    for setting in ("Trained on the pooled mixture", "Trained on each dataset alone"):
        for short, key, name in DATASETS:
            for kind, modes in KINDS.items():
                sweep = corr.load(f"mixture_{kind}") if setting.startswith("Trained on the pooled") else corr.load(f"{short}_swap_{kind}")
                if sweep is None:
                    continue
                linear, head = sweep["winners"]["old_linear"], sweep["winners"]["old"]
                for mode in modes:
                    def node(winner):
                        return winner["test_per_dataset"][key][mode] if setting.startswith("Trained on the pooled") else winner["test"][mode]
                    row = {"setting": setting, "dataset": name, "kind": kind, "mode": mode, "head_label": LABEL[(kind, mode)], "token_auroc": token_auroc(key),
                           "linear_arch": linear["config"]["architecture"], "head_arch": head["config"]["architecture"],
                           "linear_params": params(kind, linear["config"]["architecture"]), "head_params": params(kind, head["config"]["architecture"])}
                    for metric in METRICS:
                        row[f"linear_{metric}"], row[f"linear_{metric}_std"] = pick(node(linear), metric)
                        row[f"head_{metric}"], row[f"head_{metric}_std"] = pick(node(head), metric)
                        row[f"diff_{metric}"] = row[f"head_{metric}"] - row[f"linear_{metric}"]
                    rows.append(row)
    return rows


def verdict(row: dict) -> str:
    """Whether the AUROC difference is larger than the combined seed spread."""
    spread = (row["linear_correctness_auroc_std"] ** 2 + row["head_correctness_auroc_std"] ** 2) ** 0.5
    if row["head_arch"] == "linear":
        return "same model (the best head is linear)"
    if abs(row["diff_correctness_auroc"]) <= spread:
        return "within seed spread"
    return "head better" if row["diff_correctness_auroc"] > 0 else "linear probe better"


def markdown(rows: list[dict]) -> str:
    """The comparison as markdown tables, one per setting."""
    out = ["# Trained heads against linear probes, per dataset", "",
           "Linear probe: the best linear head of the sweep. Trained head: the best head of the same sweep over linear, MLP-256 and MLP-512. Both chosen on validation with the old loss; "
           "test numbers are the mean over three seeds. AUROC difference is head minus linear probe. The untrained answer-token probability is the reference.", ""]
    for setting in ("Trained on the pooled mixture", "Trained on each dataset alone"):
        out += [f"## {setting}", "", "| Dataset | Head | Answer-token prob. AUROC | Linear probe AUROC | Trained head AUROC | Difference | Head architecture (parameters) | Linear probe error AUPRC | Trained head error AUPRC | Reading |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["setting"] != setting:
                continue
            out.append(f"| {r['dataset']} | {r['head_label']} | {r['token_auroc']:.3f} | {r['linear_correctness_auroc']:.3f} ± {r['linear_correctness_auroc_std']:.3f} | {r['head_correctness_auroc']:.3f} ± {r['head_correctness_auroc_std']:.3f} | "
                       f"{r['diff_correctness_auroc']:+.3f} | {r['head_arch']} ({r['head_params']:,}) | {r['linear_error_auprc']:.3f} | {r['head_error_auprc']:.3f} | {verdict(r)} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    """Write the markdown comparison next to the other results."""
    rows = comparison_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(markdown(rows), encoding="utf-8")
    print(f"[table] wrote {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
