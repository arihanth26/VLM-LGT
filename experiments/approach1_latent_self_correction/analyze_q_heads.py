# analyze_q_heads.py
#
# Post-sweep analysis for one dataset. Reloads the winning heads saved by the
# sweeps, scores the test split with every scorer (token confidence, separate
# pre-generation head, separate verifier, unified head in both modes), and
# writes the curves the report plots: accuracy when the least trusted answers
# are dropped, how many errors the lowest-scored answers contain, reliability
# bins, score histograms, accuracy by question category, and a few case
# studies (errors caught, errors missed, false alarms) with one thumbnail
# per group. Unified curves
# are computed per seed and then averaged, so they match the sweep's
# mean-over-seeds test numbers.

import argparse
import base64
import io
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.sweep_q_heads import make_head
from experiments.approach1_latent_self_correction.train_q_heads import average_precision, metrics, roc_auc
from experiments.approach1_latent_self_correction.unified_q_head import (
    UnifiedQHead,
    mode_inputs,
    predict_logits,
    to_probabilities,
)

HERE = Path(__file__).parent
# Raw datasets live under data/, not here, this script only reads them.
RAW_DIR = HERE.parents[1] / "data" / "approach1_latent_self_correction" / "raw"
SCORERS = (
    "token_confidence",
    "separate_pre_generation",
    "separate_verifier",
    "unified_pre_generation",
    "unified_joint",
)
GRID = [round(value, 2) for value in np.arange(0.05, 1.0001, 0.05)]


def tie_broken_order(scores: np.ndarray, descending: bool, seed: int = 0) -> np.ndarray:
    """Sort indices by score, breaking exact ties at random so tied scores carry no bias."""
    noise = np.random.default_rng(seed).random(len(scores))
    key = -scores if descending else scores
    return np.lexsort((noise, key))


def accuracy_at_coverage(labels: np.ndarray, scores: np.ndarray) -> list[float]:
    """Accuracy over the most trusted fraction of answers, for each coverage in the grid."""
    order = tie_broken_order(scores, descending=True)
    return [float(labels[order[: max(1, math.ceil(len(labels) * c))]].mean()) for c in GRID]


def error_capture(labels: np.ndarray, scores: np.ndarray) -> list[float]:
    """Share of all errors found inside the least trusted fraction of answers."""
    order = tie_broken_order(scores, descending=False)
    errors = 1.0 - labels
    return [
        float(errors[order[: max(1, math.ceil(len(labels) * f))]].sum() / errors.sum()) for f in GRID
    ]


def reliability(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    """Mean predicted probability against observed accuracy in equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        last = index == bins - 1
        mask = (probabilities >= low) & ((probabilities <= high) if last else (probabilities < high))
        if mask.any():
            rows.append(
                {
                    "low": float(low),
                    "count": int(mask.sum()),
                    "mean_probability": float(probabilities[mask].mean()),
                    "accuracy": float(labels[mask].mean()),
                }
            )
    return rows


def histogram(labels: np.ndarray, probabilities: np.ndarray, bins: int = 20) -> dict:
    """Counts of correct and wrong answers per predicted-probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    return {
        "edges": edges.tolist(),
        "correct": np.histogram(probabilities[labels == 1], edges)[0].tolist(),
        "wrong": np.histogram(probabilities[labels == 0], edges)[0].tolist(),
    }


def paired_bootstrap(labels: np.ndarray, first: np.ndarray, second: np.ndarray, resamples: int = 500) -> dict:
    """Difference in AUROC and error AUPRC (first minus second) with a 95% paired bootstrap interval."""
    rng = np.random.default_rng(0)
    scorers = {
        "auroc": lambda y, p: roc_auc(y, p),
        "error_auprc": lambda y, p: average_precision(1 - y, 1 - p),
    }
    result = {}
    for name, function in scorers.items():
        point = function(labels, first) - function(labels, second)
        diffs = []
        for _ in range(resamples):
            rows = rng.integers(0, len(labels), len(labels))
            if labels[rows].min() == labels[rows].max():
                continue
            diffs.append(function(labels[rows], first[rows]) - function(labels[rows], second[rows]))
        low, high = np.percentile(diffs, [2.5, 97.5])
        result[name] = {"difference": float(point), "ci_low": float(low), "ci_high": float(high)}
    return result


def separate_probabilities(folder: Path, test: dict, device: torch.device) -> dict[str, np.ndarray]:
    """Score the test split with the best separate pre-generation head and verifier."""
    results = json.loads((folder / "sweep_results.json").read_text(encoding="utf-8"))
    outputs = {}
    for name, key, feature_key in (
        ("pre_generation", "separate_pre_generation", "prompt_features"),
        ("candidate_verifier", "separate_verifier", "generated_features"),
    ):
        features = test[feature_key]
        head = make_head(results[name]["best_config"]["architecture"], features.shape[1]).to(device)
        head.load_state_dict(torch.load(folder / f"sweep_best_{name}.pt", map_location=device))
        head.eval()
        with torch.inference_mode():
            logits = torch.cat(
                [head(features[i : i + 1024].to(device)).cpu() for i in range(0, len(features), 1024)]
            ).numpy()
        outputs[key] = to_probabilities(logits, results[name]["test"]["temperature"])
    return outputs


def unified_probabilities(folder: Path, test: dict, device: torch.device) -> dict[str, list[np.ndarray]]:
    """Score the test split with each seed of the unified winner, in both modes."""
    results = json.loads((folder / "unified_sweep_results.json").read_text(encoding="utf-8"))
    winner = results["winner"]
    size = test["prompt_features"].shape[1]
    outputs = {"unified_joint": [], "unified_pre_generation": []}
    for seed in (1, 2, 3):
        run = next(r for r in results["runs"] if r["config_id"] == winner["config_id"] and r["seed"] == seed)
        head = UnifiedQHead(size, winner["config"]["architecture"]).to(device)
        head.load_state_dict(torch.load(folder / f"unified_winner_seed_{seed}.pt", map_location=device))
        for mode, key in (("joint", "unified_joint"), ("pre_generation", "unified_pre_generation")):
            prompt, answer = mode_inputs(test, mode)
            logits = predict_logits(head, prompt, answer, device)
            outputs[key].append(to_probabilities(logits, run["temperatures"][mode]))
    return outputs


def load_records(baseline_path: Path) -> dict[int, dict]:
    """Index the successful baseline records by dataset index."""
    records = {}
    for line in baseline_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "prediction" in record:
                records[record["index"]] = record
    return records


def token_confidence(records: dict[int, dict], indices: list[int]) -> np.ndarray:
    """Geometric-mean token confidence of each test answer, in feature order."""
    return np.array([records[index]["prediction"]["confidence"] for index in indices])


def thumbnail(raw_dir: Path, dataset: str, index: int, max_side: int) -> str | None:
    """Return a small base64 JPEG of one test image, or None if it cannot be loaded."""
    from datasets import load_from_disk

    from experiments.approach1_latent_self_correction.baseline import extract_image

    try:
        folder = raw_dir / dataset
        example = load_from_disk(str(folder))["test"][int(index)]
        image = extract_image(example, folder)
    except Exception:
        return None
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=72)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def case_studies(
    dataset: str, raw_dir: Path, records: dict, indices: list[int], labels: np.ndarray,
    joint: np.ndarray, pre: np.ndarray, confidence: np.ndarray, categories: list[str],
) -> dict:
    """Pick a few examples per outcome group, with scores, and one thumbnail per group."""
    wrong = np.where(labels == 0)[0]
    right = np.where(labels == 1)[0]
    groups = {
        "caught_errors": wrong[np.argsort(joint[wrong])][:3],
        "missed_errors": wrong[np.argsort(-joint[wrong])][:3],
        "false_alarms": right[np.argsort(joint[right])][:3],
        "confident_correct": right[np.argsort(-joint[right])][:3],
    }
    max_side = 640 if dataset == "docvqa" else 420
    result = {}
    for name, rows in groups.items():
        items = []
        for rank, row in enumerate(rows):
            record = records[indices[row]]
            items.append(
                {
                    "index": int(indices[row]),
                    "question": str(record["question"])[:420],
                    "reference": str(record["reference_answer"])[:120],
                    "prediction": str(record["prediction"]["answer"])[:120],
                    "category": categories[row] if categories else "",
                    "joint_score": float(joint[row]),
                    "pre_score": float(pre[row]),
                    "token_confidence": float(confidence[row]),
                    "image": thumbnail(raw_dir, dataset, indices[row], max_side) if rank == 0 else None,
                }
            )
        result[name] = items
    return result


def category_labels(raw_dir: Path, dataset: str, indices: list[int]) -> list[str]:
    """Category of each test example in feature order, or an empty list if the dataset has none."""
    from datasets import load_from_disk

    split = load_from_disk(str(raw_dir / dataset))["test"]
    if "category" not in split.column_names:
        return []
    column = split["category"]
    return [column[index] for index in indices]


def category_table(names: list[str], labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Accuracy and mean predicted correctness per question category, when the dataset has one."""
    if not names:
        return []
    categories = np.array(names)
    rows = []
    for name in sorted(set(categories)):
        mask = categories == name
        if name and mask.sum() >= 30:
            rows.append(
                {
                    "category": name,
                    "count": int(mask.sum()),
                    "accuracy": float(labels[mask].mean()),
                    "mean_predicted": float(scores[mask].mean()),
                }
            )
    return sorted(rows, key=lambda row: -row["count"])


def split_summary(feature_dir: Path, dataset: str, size: str) -> dict:
    """Example count and base-model accuracy for each cached split."""
    summary = {}
    for split in ("train", "val", "test"):
        payload = torch.load(feature_dir / f"{dataset}_{split}_{size}.pt", map_location="cpu", weights_only=True)
        summary[split] = {"count": int(len(payload["labels"])), "accuracy": float(payload["labels"].mean())}
    return summary


def main() -> None:
    """Parse arguments, score the test split with every scorer, and write the analysis JSON."""
    parser = argparse.ArgumentParser(description="Analyze trained Q-heads on one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--size", default="2b")
    args = parser.parse_args()

    feature_dir = HERE / "q_features"
    test = torch.load(feature_dir / f"{args.dataset}_test_{args.size}.pt", map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = test["labels"].numpy()
    indices = test["indices"].tolist()

    records = load_records(HERE / "results" / f"{args.dataset}_test_{args.size}_baseline.jsonl")
    per_scorer = {"token_confidence": [token_confidence(records, indices)]}
    folder = HERE / "q_models" / args.dataset
    for key, values in separate_probabilities(folder, test, device).items():
        per_scorer[key] = [values]
    per_scorer.update(unified_probabilities(folder, test, device))

    analysis = {
        "dataset": args.dataset,
        "splits": split_summary(feature_dir, args.dataset, args.size),
        "grid": GRID,
        "scorers": {},
    }
    for name in SCORERS:
        runs = per_scorer[name]
        analysis["scorers"][name] = {
            "accuracy_at_coverage": np.mean([accuracy_at_coverage(labels, run) for run in runs], axis=0).tolist(),
            "error_capture": np.mean([error_capture(labels, run) for run in runs], axis=0).tolist(),
        }
    mean_joint = np.mean(per_scorer["unified_joint"], axis=0)
    analysis["reliability"] = {
        "token_confidence": reliability(labels, per_scorer["token_confidence"][0]),
        "unified_joint": reliability(labels, mean_joint),
    }
    analysis["histogram_unified_joint"] = histogram(labels, mean_joint)
    mean_pre = np.mean(per_scorer["unified_pre_generation"], axis=0)
    analysis["bootstrap"] = {
        "joint_minus_separate_verifier": paired_bootstrap(labels, mean_joint, per_scorer["separate_verifier"][0]),
        "unified_pre_minus_separate_pre": paired_bootstrap(labels, mean_pre, per_scorer["separate_pre_generation"][0]),
    }
    analysis["token_confidence_metrics"] = metrics(labels, per_scorer["token_confidence"][0])
    lengths = np.array([len(str(records[index]["prediction"]["answer"])) for index in indices], dtype=float)
    analysis["answer_length_baseline"] = {
        "auroc": float(roc_auc(labels, -lengths)),
        "error_auprc": float(average_precision(1 - labels, lengths)),
        "mean_length_correct": float(lengths[labels == 1].mean()),
        "mean_length_wrong": float(lengths[labels == 0].mean()),
    }
    analysis["token_confidence_exactly_one"] = float((per_scorer["token_confidence"][0] >= 0.9999).mean())
    names = category_labels(RAW_DIR, args.dataset, indices)
    analysis["categories"] = category_table(names, labels, mean_joint)
    analysis["case_studies"] = case_studies(
        args.dataset, RAW_DIR, records, indices, labels, mean_joint, mean_pre,
        per_scorer["token_confidence"][0], names,
    )

    output = HERE / "results" / f"analysis_{args.dataset}.json"
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {output}")


if __name__ == "__main__":
    main()
