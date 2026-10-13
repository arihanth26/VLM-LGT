# analyze_mixture.py
#
# Post-sweep analysis of the dataset-mixture experiments. Scores the mixture test
# split with the model's own confidence and an answer-length rule, loads the saved
# test scores of the winning heads from sweep_mixture.py, and writes what the
# report plots: accuracy when the least trusted answers are dropped and the share
# of errors found at each flag rate (pooled and per dataset), the grounding
# metrics of the model's own confidence, the mixture composition, and a gallery of
# clean versus corrupted images. Output: results/analysis_mixture.json.

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.analyze_q_heads import (
    GRID,
    accuracy_at_coverage,
    error_capture,
)
from experiments.approach1_latent_self_correction.corruption import corrupt_image, swap_partners
from experiments.approach1_latent_self_correction.mixture_heads import (
    MIN_DATASET_ROWS,
    grounding_metrics,
    key_metrics,
)
from experiments.approach1_latent_self_correction.train_q_heads import average_precision, roc_auc

HERE = Path(__file__).parent
FEATURE_DIR = HERE / "q_features"
MODELS_DIR = HERE / "q_models" / "mixture"
RAW_DIR = HERE.parents[1] / "data" / "approach1_latent_self_correction" / "raw"
KINDS = ("pre", "verifier", "unified")
CURVE_WINNERS = ("old", "new", "old_linear", "new_linear")


def per_dataset_rows(labels: np.ndarray, dataset_id: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Row masks for each dataset with enough examples of both classes."""
    rows = {"pooled": np.ones_like(labels, dtype=bool)}
    for index, name in enumerate(names):
        mask = dataset_id == index
        if mask.sum() >= MIN_DATASET_ROWS and 0 < labels[mask].mean() < 1:
            rows[name] = mask
    return rows


def thumbnail(image, max_side: int = 360) -> str:
    """A small base64 JPEG of a PIL image."""
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=72)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def gallery(names: list[str], test: dict, size: str) -> list[dict]:
    """One clean and one corrupted image per dataset, rebuilt with the same partner rule as extraction."""
    from datasets import load_from_disk

    from experiments.approach1_latent_self_correction.baseline import extract_image

    entries = []
    for index, name in enumerate(names):
        try:
            folder = RAW_DIR / name
            split = load_from_disk(str(folder))["test"]
            partners = swap_partners(len(split), 0)
            row = int(np.where(test["dataset_id"].numpy() == index)[0][3])
            dataset_index = int(test["indices"][row])
            clean = extract_image(split[dataset_index], folder)
            partner = extract_image(split[partners[dataset_index]], folder)
            entries.append(
                {
                    "dataset": name,
                    "clean": thumbnail(clean),
                    "corrupt": thumbnail(corrupt_image(clean, "swap", 0, partner)),
                    "question": str(split[dataset_index].get("question", ""))[:200],
                }
            )
        except Exception as error:  # a missing raw dataset must not block the rest of the analysis
            print(f"[skip gallery] {name}: {error}")
    return entries


def main() -> None:
    """Build the analysis file for the mixture experiments."""
    parser = argparse.ArgumentParser(description="Analyze the dataset-mixture experiments.")
    parser.add_argument("--size", default="2b")
    parser.add_argument("--tag", default="mixture")
    args = parser.parse_args()

    splits = {s: torch.load(FEATURE_DIR / f"{args.tag}_{s}_{args.size}.pt", map_location="cpu", weights_only=True) for s in ("train", "val", "test")}
    test = splits["test"]
    names = list(test["dataset_names"])
    labels = test["labels"].numpy()
    dataset_id = test["dataset_id"].numpy()
    rows = per_dataset_rows(labels, dataset_id, names)

    analysis = {"datasets": names, "grid": GRID, "composition": {}, "baselines": {}, "curves": {}}
    for split, data in splits.items():
        analysis["composition"][split] = {
            "rows": int(len(data["labels"])),
            "accuracy": float(data["labels"].mean()),
            "per_dataset": {
                name: {"rows": int((data["dataset_id"] == i).sum()), "accuracy": float(data["labels"][data["dataset_id"] == i].mean())}
                for i, name in enumerate(names)
            },
        }

    confidence = test["token_confidence"].numpy()
    tf_clean, tf_corrupt = test["confidence_clean"].numpy(), test["confidence_corrupt"].numpy()
    length = test["answer_length"].numpy()
    baselines = {}
    for label, mask in rows.items():
        entry = {
            "token_confidence": key_metrics(labels[mask], confidence[mask]),
            "exactly_one": float((confidence[mask] >= 0.9999).mean()),
            "answer_length": {
                "auroc": float(roc_auc(labels[mask], -length[mask])),
                "error_auprc": float(average_precision(1 - labels[mask], length[mask])),
            },
            "grounding_token_confidence": grounding_metrics(tf_clean[mask], tf_corrupt[mask], labels[mask]),
        }
        baselines[label] = entry
    analysis["baselines"] = baselines
    analysis["token_curves"] = {
        label: {
            "accuracy_at_coverage": accuracy_at_coverage(labels[mask], confidence[mask]),
            "error_capture": error_capture(labels[mask], confidence[mask]),
        }
        for label, mask in rows.items()
    }

    for kind in KINDS:
        path = MODELS_DIR / f"{args.tag}_{kind}_scores.pt"
        if not path.exists():
            continue
        saved = torch.load(path, map_location="cpu", weights_only=True)
        analysis["curves"][kind] = {}
        for winner in CURVE_WINNERS:
            if winner not in saved["scores"]:
                continue
            analysis["curves"][kind][winner] = {}
            for mode, sides in saved["scores"][winner].items():
                clean = sides["clean"].float().numpy()
                per_group = {}
                for label, mask in rows.items():
                    if label != "pooled" and winner not in ("old", "new"):
                        continue
                    per_group[label] = {
                        "accuracy_at_coverage": np.mean([accuracy_at_coverage(labels[mask], run[mask]) for run in clean], axis=0).tolist(),
                        "error_capture": np.mean([error_capture(labels[mask], run[mask]) for run in clean], axis=0).tolist(),
                    }
                analysis["curves"][kind][winner][mode] = per_group
    analysis["gallery"] = gallery(names, test, args.size)

    output = HERE / "results" / "analysis_mixture.json"
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {output}")


if __name__ == "__main__":
    main()
