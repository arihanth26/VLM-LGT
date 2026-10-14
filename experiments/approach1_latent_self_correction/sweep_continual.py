# sweep_continual.py
#
# Runs the sequential-training experiments for one head kind. Each head is trained
# through several dataset orders, once per loss family (old loss: no grounding
# term, new loss: with it) and with or without a small replay buffer of earlier
# data, on three seeds. It reports the stage-by-dataset score matrices, the
# retention metrics from continual.py averaged over orders and seeds, and the score
# of a head trained on each dataset alone as the reference. The head configuration
# of each loss family is the winner of the mixture sweep for the same kind, so the
# losses are compared at the settings that were chosen on validation.
# Output: q_models/mixture/continual_<kind>_results.json.

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.continual import (
    run_sequence,
    sequence_metrics,
    split_by_dataset,
    train_isolated,
    matrix_of,
)
from experiments.approach1_latent_self_correction.mixture_heads import HeadSpec, modes_for
from experiments.approach1_latent_self_correction.sweep_mixture import FEATURE_DIR, MODELS_DIR, load_split

SEEDS = (1, 2, 3)
ORDER_NAMES = {
    "forward": ["chartqa_full", "textvqa", "docvqa", "scienceqa_img"],
    "reverse": ["scienceqa_img", "docvqa", "textvqa", "chartqa_full"],
    "shuffled": ["docvqa", "scienceqa_img", "chartqa_full", "textvqa"],
}
DEFAULT_CONFIG = {
    "pre": {"architecture": "mlp_512", "learning_rate": 1e-3, "rank_weight": 0.0},
    "verifier": {"architecture": "mlp_512", "learning_rate": 1e-3, "pair_weight": 0.0},
    "unified": {"architecture": "mlp_512", "learning_rate": 1e-3, "pre_weight": 1.0, "pair_weight": 0.0},
}


def stats(values: list[float]) -> dict:
    """Mean and sample standard deviation of a list of numbers."""
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def load_config(kind: str, loss: str, results_dir: Path, tag: str) -> dict:
    """The head configuration of one loss family: the mixture-sweep winner, or a default."""
    path = results_dir / f"{tag}_{kind}_results.json"
    if path.exists():
        winners = json.loads(path.read_text(encoding="utf-8"))["winners"]
        if loss in winners:
            config = dict(winners[loss]["config"])
            config.pop("seed", None)
            return config
    config = {**HeadSpec(kind, "mlp_512", 1e-3).config(), **DEFAULT_CONFIG[kind]}
    if loss == "new":
        config["ground_weight"] = 0.25
    return config


def run_variant(
    kind: str, config: dict, orders: dict, train: list, validation: list, test: list, names: list[str],
    replay: float, device: torch.device, max_epochs: int, patience: int, batch_size: int,
) -> dict:
    """All orders and seeds of one (loss, replay) variant, plus the isolated reference."""
    modes = modes_for(kind)
    isolated = {mode: {d: [] for d in range(len(names))} for mode in modes}
    for seed in SEEDS:
        spec = HeadSpec(**{**config, "kind": kind, "seed": seed})
        for dataset in range(len(names)):
            scores = train_isolated(spec, train[dataset], validation[dataset], test, device, max_epochs, patience, batch_size)
            for mode in modes:
                isolated[mode][dataset].append(scores[dataset][mode]["auroc"])
    iso_mean = {mode: np.array([np.mean(isolated[mode][d]) for d in range(len(names))]) for mode in modes}

    result = {"config": config, "replay": replay, "orders": {}, "isolated": {
        mode: {names[d]: stats(isolated[mode][d]) for d in range(len(names))} for mode in modes}}
    pooled = {mode: {metric: [] for metric in ("final_avg", "forgetting", "bwt", "retention")} for mode in modes}
    for order_name, order_names in orders.items():
        order = [names.index(n) for n in order_names]
        runs = []
        for seed in SEEDS:
            runs.append(run_sequence(HeadSpec(**{**config, "kind": kind, "seed": seed}), order, train, validation, test, device, replay, max_epochs, patience, batch_size))
            print(f"[{kind} replay={replay} order={order_name} seed={seed}] done", flush=True)
        entry = {"order": order_names, "matrix": {}, "grounding_matrix": {}, "metrics": {}}
        for mode in modes:
            matrices = [matrix_of(run, mode, "auroc") for run in runs]
            entry["matrix"][mode] = np.mean(matrices, axis=0).tolist()
            entry["grounding_matrix"][mode] = np.mean([matrix_of(run, mode, "order_accuracy") for run in runs], axis=0).tolist()
            per_seed = [sequence_metrics(m, order, iso_mean[mode]) for m in matrices]
            entry["metrics"][mode] = {k: stats([row[k] for row in per_seed]) for k in per_seed[0]}
            for row in per_seed:
                for metric, value in row.items():
                    pooled[mode][metric].append(value)
        result["orders"][order_name] = entry
    result["summary"] = {mode: {metric: stats(values) for metric, values in pooled[mode].items()} for mode in modes}
    return result


def main() -> None:
    """Parse arguments and run every loss and replay variant for one head kind."""
    parser = argparse.ArgumentParser(description="Sequential training across datasets for one head kind.")
    parser.add_argument("--kind", choices=("pre", "verifier", "unified"), required=True)
    parser.add_argument("--tag", default="mixture")
    parser.add_argument("--size", default="2b")
    parser.add_argument("--losses", nargs="+", default=["old", "new"])
    parser.add_argument("--replay", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--orders", nargs="+", default=list(ORDER_NAMES))
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out-tag", default=None, help="Name for the output file, default the input tag.")
    args = parser.parse_args()
    out_tag = args.out_tag or args.tag

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_all, names = load_split(FEATURE_DIR / f"{args.tag}_train_{args.size}.pt", device)
    validation_all, _ = load_split(FEATURE_DIR / f"{args.tag}_val_{args.size}.pt", device)
    test_all, _ = load_split(FEATURE_DIR / f"{args.tag}_test_{args.size}.pt", device)
    train, validation, test = (split_by_dataset(split, len(names)) for split in (train_all, validation_all, test_all))
    orders = {name: ORDER_NAMES[name] for name in args.orders}

    output = {"kind": args.kind, "datasets": names, "seeds": list(SEEDS), "variants": {}}
    for loss in args.losses:
        config = load_config(args.kind, loss, args.output_dir, args.tag)
        for replay in args.replay:
            key = f"{loss}_replay{replay:g}"
            output["variants"][key] = run_variant(
                args.kind, config, orders, train, validation, test, names, replay, device,
                args.max_epochs, args.patience, args.batch_size,
            )
            summary = output["variants"][key]["summary"]
            mode = modes_for(args.kind)[0]
            print(f"[{args.kind}] {key}: final {summary[mode]['final_avg']['mean']:.3f} forgetting {summary[mode]['forgetting']['mean']:.3f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"continual_{args.kind}_results.json" if out_tag == "mixture" else f"{out_tag}_continual_{args.kind}_results.json"
    (args.output_dir / name).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
