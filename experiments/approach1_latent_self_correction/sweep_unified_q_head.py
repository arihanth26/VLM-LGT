# sweep_unified_q_head.py
#
# Sweeps the unified Q-head (see unified_q_head.py) over cached Qwen3-VL
# features for one dataset. Follows the protocol of sweep_verifier_early_stop.py:
# every config runs on 3 seeds with early stopping on validation, the winning
# config is the one with the best mean validation score, and the test split is
# touched once, for that winner only. Both modes (joint and pre-generation)
# are reported so the result can be put next to the separate heads from
# sweep_q_heads.py.

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.unified_q_head import (
    MODES,
    UnifiedConfig,
    UnifiedQHead,
    evaluate_modes,
    train_unified,
)

SEEDS = (1, 2, 3)
SUMMARY_METRICS = ("correctness_auroc", "error_auprc", "brier", "ece_10_bin", "pearson_correctness")


def build_grid(quick: bool, rank_weight: float) -> list[dict]:
    """List every config in the sweep, without the seed."""
    if quick:
        axes = (("linear", "mlp_512"), (1e-3,), (0.0, 1.0), (0.0,))
    else:
        axes = (("linear", "mlp_256", "mlp_512"), (3e-4, 1e-3), (0.0, 0.5, 1.0), (0.0, 0.25))
    return [
        {
            "architecture": architecture,
            "learning_rate": learning_rate,
            "pre_weight": pre_weight,
            "rank_weight": rank_weight,
            "pair_weight": pair_weight,
        }
        for architecture, learning_rate, pre_weight, pair_weight in itertools.product(*axes)
    ]


def mean_std(values: list[float]) -> dict:
    """Mean and sample standard deviation of a short list of run values."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def run_sweep(
    train: dict,
    validation: dict,
    test: dict,
    grid: list[dict],
    device: torch.device,
    max_epochs: int,
    patience: int,
    select: str,
) -> tuple[dict, dict]:
    """Train the whole grid, pick the winner on validation, and score it on test."""
    runs = []
    states = {}
    for number, params in enumerate(grid, start=1):
        for seed in SEEDS:
            config = UnifiedConfig(**params, seed=seed)
            best = train_unified(train, validation, config, device, max_epochs, patience, select)
            states[(number, seed)] = best.pop("state")
            runs.append({"config_id": number, "config": params, "seed": seed, **best})
            print(
                f"[config {number}/{len(grid)} seed={seed}] {params} epoch={best['epoch']} "
                + " ".join(
                    f"{mode}_error_auprc={best['validation'][mode]['error_auprc']:.4f}" for mode in MODES
                ),
                flush=True,
            )

    aggregates = []
    for number, params in enumerate(grid, start=1):
        chosen = [run for run in runs if run["config_id"] == number]
        aggregates.append(
            {
                "config_id": number,
                "config": params,
                "mean_val_error_auprc": {
                    mode: float(np.mean([run["validation"][mode]["error_auprc"] for run in chosen]))
                    for mode in MODES
                },
                "mean_val_auroc": {
                    mode: float(np.mean([run["validation"][mode]["correctness_auroc"] for run in chosen]))
                    for mode in MODES
                },
            }
        )

    def score(item: dict) -> tuple[float, float]:
        picked = MODES if select == "mean" else (select,)
        return (
            float(np.mean([item["mean_val_error_auprc"][mode] for mode in picked])),
            float(np.mean([item["mean_val_auroc"][mode] for mode in picked])),
        )

    winner = max(aggregates, key=score)
    feature_size = train["prompt_features"].shape[1]
    test_runs = []
    winner_states = {}
    for seed in SEEDS:
        run = next(r for r in runs if r["config_id"] == winner["config_id"] and r["seed"] == seed)
        head = UnifiedQHead(feature_size, winner["config"]["architecture"]).to(device)
        head.load_state_dict(states[(winner["config_id"], seed)])
        test_runs.append(
            {"seed": seed, "best_epoch": run["epoch"], **evaluate_modes(head, run["temperatures"], test, device)}
        )
        winner_states[seed] = states[(winner["config_id"], seed)]

    test_summary = {
        mode: {
            name: mean_std([run[mode][name] for run in test_runs]) for name in SUMMARY_METRICS
        }
        for mode in MODES
    }
    result = {
        "select": select,
        "max_epochs": max_epochs,
        "patience": patience,
        "winner": winner,
        "aggregates": aggregates,
        "runs": runs,
        "test_runs": test_runs,
        "test_summary": test_summary,
    }
    return result, winner_states


def main() -> None:
    """Parse arguments, run the sweep, and write results and winner checkpoints."""
    parser = argparse.ArgumentParser(description="Sweep the unified Q-head on cached features.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--select", choices=("mean", "joint", "pre_generation"), default="mean")
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true", help="Run a small grid for a smoke test.")
    args = parser.parse_args()

    train = torch.load(args.train, map_location="cpu", weights_only=True)
    validation = torch.load(args.validation, map_location="cpu", weights_only=True)
    test = torch.load(args.test, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result, winner_states = run_sweep(
        train,
        validation,
        test,
        build_grid(args.quick, args.rank_weight),
        device,
        args.max_epochs,
        args.patience,
        args.select,
    )
    result["dataset"] = train.get("dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for seed, state in winner_states.items():
        torch.save(state, args.output.with_name(f"unified_winner_seed_{seed}.pt"))
    print(json.dumps({"winner": result["winner"], "test_summary": result["test_summary"]}, indent=2))


if __name__ == "__main__":
    main()
