# sweep_mixture.py
#
# Sweeps one head kind (pre, verifier, or unified) on the dataset mixture, under
# the old loss (ground_weight = 0) and the new loss (ground_weight > 0). Every
# configuration is trained on three seeds with early stopping on validation, the
# winner of each group is the configuration with the best mean validation score,
# and test numbers are read only for those winners. Groups reported:
#   old / new            best configuration of each loss family
#   new_full             new loss with both contrastive terms on (pair and ground)
#   old_linear / new_linear   best linear probe of each family
#   cell_*               a 2 x 2 ablation of the pair term against the ground term
# Test results are pooled and per dataset, with grounding metrics.

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.mixture_heads import (
    HeadSpec,
    modes_for,
    selection_key,
    to_device,
    train_head,
)

FEATURE_DIR = Path(__file__).parent / "q_features"
MODELS_DIR = Path(__file__).parent / "q_models" / "mixture"
SEEDS = (1, 2, 3)


def build_grid(kind: str, quick: bool) -> list[HeadSpec]:
    """List every configuration for a head kind, without the seed."""
    architectures = ("linear", "mlp_256") if quick else ("linear", "mlp_256", "mlp_512")
    rates = (1e-3,) if quick else (3e-4, 1e-3)
    grounds = (0.0, 0.5) if quick else (0.0, 0.1, 0.5)
    specs = []
    if kind == "pre":
        for arch, rate, rank, ground in itertools.product(architectures, rates, (0.0, 0.25), grounds):
            specs.append(HeadSpec(kind, arch, rate, rank_weight=rank, ground_weight=ground))
    elif kind == "verifier":
        for arch, rate, pair, ground in itertools.product(architectures, rates, (0.0, 0.25), grounds):
            specs.append(HeadSpec(kind, arch, rate, pair_weight=pair, ground_weight=ground))
    elif kind == "unified":
        for arch, rate, pre, pair, ground in itertools.product(architectures, rates, (0.0, 1.0), (0.0, 0.25), grounds):
            specs.append(HeadSpec(kind, arch, rate, pre_weight=pre, pair_weight=pair, ground_weight=ground))
    else:
        raise ValueError(f"unknown head kind '{kind}'")
    return specs


def aggregate(items: list):
    """Mean and standard deviation over seeds for a list of nested dicts of numbers."""
    first = items[0]
    if isinstance(first, dict):
        return {key: aggregate([item[key] for item in items]) for key in first if all(key in item for item in items)}
    values = np.array(items, dtype=float)
    return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0}


def mean_only(node):
    """Collapse an aggregate tree to its means."""
    if isinstance(node, dict) and set(node) == {"mean", "std"}:
        return node["mean"]
    return {key: mean_only(value) for key, value in node.items()}


def config_id(config: dict) -> str:
    """A stable text key for a configuration."""
    return json.dumps(config, sort_keys=True)


def group_runs(runs: list[dict]) -> dict:
    """Group seed runs by configuration and attach their mean validation score."""
    groups: dict = {}
    for run in runs:
        groups.setdefault(config_id({k: v for k, v in run["spec"].items() if k != "seed"}), []).append(run)
    summary = {}
    for key, members in groups.items():
        val = aggregate([run["val_pooled"] for run in members])
        kind = members[0]["spec"]["kind"]
        summary[key] = {"config": json.loads(key), "runs": members, "val": val, "score": selection_key(mean_only(val), kind)}
    return summary


def predicates(kind: str) -> dict:
    """The winner groups to report, as filters over a configuration dict."""
    groups = {
        "old": lambda c: c["ground_weight"] == 0,
        "new": lambda c: c["ground_weight"] > 0,
        "old_linear": lambda c: c["ground_weight"] == 0 and c["architecture"] == "linear",
        "new_linear": lambda c: c["ground_weight"] > 0 and c["architecture"] == "linear",
    }
    if kind != "pre":
        groups["new_full"] = lambda c: c["ground_weight"] > 0 and c["pair_weight"] > 0
        groups["cell_plain"] = lambda c: c["ground_weight"] == 0 and c["pair_weight"] == 0
        groups["cell_contrast"] = lambda c: c["ground_weight"] == 0 and c["pair_weight"] > 0
        groups["cell_ground"] = lambda c: c["ground_weight"] > 0 and c["pair_weight"] == 0
        groups["cell_full"] = lambda c: c["ground_weight"] > 0 and c["pair_weight"] > 0
    return groups


def pack_winner(group: dict, names: list[str]) -> tuple[dict, dict]:
    """Aggregate a winning configuration's seed runs into a result entry and a scores entry."""
    runs = group["runs"]
    kind = runs[0]["spec"]["kind"]
    entry = {
        "config": group["config"],
        "val": group["val"],
        "epochs": [run["epoch"] for run in runs],
        "test": aggregate([run["test"]["pooled"] for run in runs]),
        "test_per_dataset": aggregate([run["test"]["per_dataset"] for run in runs]),
        "grounding": aggregate([run["test"]["grounding"] for run in runs]),
    }
    scores = {}
    for mode in modes_for(kind):
        scores[mode] = {
            side: torch.from_numpy(np.stack([run["test"]["probabilities"][mode][side] for run in runs]).astype(np.float16))
            for side in ("clean", "corrupt")
        }
    return entry, scores


def run_sweep(kind: str, train: dict, validation: dict, test: dict, names: list[str], device: torch.device, quick: bool,
              max_epochs: int, patience: int, batch_size: int) -> tuple[dict, dict]:
    """Train the whole grid for one head kind and pick the winner groups."""
    specs = build_grid(kind, quick)
    runs = []
    for number, base in enumerate(specs, start=1):
        for seed in SEEDS:
            spec = HeadSpec(**{**base.__dict__, "seed": seed})
            run = train_head(spec, train, validation, test, names, device, max_epochs, patience, batch_size)
            runs.append(run)
            modes = modes_for(kind)
            print(
                f"[{kind} {number}/{len(specs)} seed={seed}] {base.config()} epoch={run['epoch']} "
                + " ".join(f"val_{m}_auroc={run['val_pooled'][m]['correctness_auroc']:.4f}" for m in modes),
                flush=True,
            )
    groups = group_runs(runs)
    winners, scores = {}, {}
    for name, predicate in predicates(kind).items():
        candidates = [g for g in groups.values() if predicate(g["config"])]
        if not candidates:
            continue
        best = max(candidates, key=lambda g: g["score"])
        winners[name], scores[name] = pack_winner(best, names)
    configs = [
        {"config": g["config"], "val": mean_only(g["val"]), "score": list(g["score"])}
        for g in groups.values()
    ]
    return {"kind": kind, "winners": winners, "configs": configs, "seeds": list(SEEDS)}, scores


def load_split(path: Path, device: torch.device) -> tuple[dict, list[str]]:
    """Load a mixture file and move what the heads need onto the device."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    tensors = to_device(data, device)
    tensors["dataset_id"] = data["dataset_id"].to(device)
    return tensors, list(data["dataset_names"])


def main() -> None:
    """Parse arguments, run the sweep for one head kind, and write results and test scores."""
    parser = argparse.ArgumentParser(description="Sweep one head kind on the dataset mixture.")
    parser.add_argument("--kind", choices=("pre", "verifier", "unified"), required=True)
    parser.add_argument("--tag", default="mixture")
    parser.add_argument("--size", default="2b")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--quick", action="store_true", help="Run a small grid for a smoke test.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, names = load_split(FEATURE_DIR / f"{args.tag}_train_{args.size}.pt", device)
    validation, _ = load_split(FEATURE_DIR / f"{args.tag}_val_{args.size}.pt", device)
    test, _ = load_split(FEATURE_DIR / f"{args.tag}_test_{args.size}.pt", device)
    result, scores = run_sweep(
        args.kind, train, validation, test, names, device, args.quick, args.max_epochs, args.patience, args.batch_size
    )
    result["datasets"] = names
    result["rows"] = {"train": int(len(train["labels"])), "val": int(len(validation["labels"])), "test": int(len(test["labels"]))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.tag}_{args.kind}_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {"scores": {name: {mode: dict(sides) for mode, sides in modes.items()} for name, modes in scores.items()},
         "labels": test["labels"].cpu(), "dataset_id": test["dataset_id"].cpu(), "dataset_names": names},
        args.output_dir / f"{args.tag}_{args.kind}_scores.pt",
    )
    print(json.dumps({name: w["config"] for name, w in result["winners"].items()}, indent=2))


if __name__ == "__main__":
    main()
