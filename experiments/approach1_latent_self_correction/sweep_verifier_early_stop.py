"""Sweep verifier ranking weights with validation-based early stopping."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.approach1_latent_self_correction.train_q_heads import (
    QHead,
    evaluate_head,
    fit_temperature,
    metrics,
    ranking_loss,
)


def validation_metrics(
    head: QHead, features: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> tuple[float, dict]:
    head.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(labels), 1024):
            logits.append(head(features[start : start + 1024].to(device)).cpu())
    values = torch.cat(logits).numpy()
    temperature = fit_temperature(values, labels.numpy())
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(values / temperature, -30, 30)))
    return temperature, metrics(labels.numpy(), probabilities)


def train_run(
    train: dict,
    validation: dict,
    device: torch.device,
    ranking_weight: float,
    seed: int,
    max_epochs: int,
    patience: int,
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = QHead(train["generated_features"].shape[1], hidden_size=512).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_key = (-1.0, -1.0)
    best = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        permutation = torch.randperm(len(train["labels"]), generator=generator)
        head.train()
        losses = []
        for start in range(0, len(permutation), 512):
            indices = permutation[start : start + 512]
            features = train["generated_features"][indices].to(device)
            labels = train["labels"][indices].to(device)
            logits = head(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if ranking_weight:
                loss = loss + ranking_weight * ranking_loss(logits, labels)
                wrong = labels == 0
                if wrong.any():
                    references = train["reference_features"][indices].to(device)
                    reference_logits = head(references[wrong])
                    loss = loss + ranking_weight * F.softplus(
                        -(reference_logits - logits[wrong])
                    ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        temperature, current_metrics = validation_metrics(
            head, validation["generated_features"], validation["labels"], device
        )
        key = (current_metrics["error_auprc"], current_metrics["correctness_auroc"])
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "temperature": temperature, **current_metrics}
        )
        if key > best_key:
            best_key = key
            best = {
                "epoch": epoch,
                "temperature": temperature,
                "validation": current_metrics,
                "state": copy.deepcopy(head.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    assert best is not None
    return {
        "ranking_weight": ranking_weight,
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "temperature": best["temperature"],
        "validation": best["validation"],
        "history": history,
        "state": best["state"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train = torch.load(args.train, map_location="cpu", weights_only=True)
    validation = torch.load(args.validation, map_location="cpu", weights_only=True)
    test = torch.load(args.test, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    states = {}
    weights = (0.0, 0.01, 0.03, 0.1, 0.25)
    for ranking_weight in weights:
        for seed in (1, 2, 3):
            run = train_run(
                train, validation, device, ranking_weight, seed, args.max_epochs, args.patience
            )
            states[(ranking_weight, seed)] = run.pop("state")
            runs.append(run)
            print(
                f"[weight={ranking_weight} seed={seed}] best_epoch={run['best_epoch']} "
                f"epochs_run={run['epochs_run']} error_auprc={run['validation']['error_auprc']:.4f} "
                f"auroc={run['validation']['correctness_auroc']:.4f}",
                flush=True,
            )

    aggregates = []
    for ranking_weight in weights:
        selected = [run for run in runs if run["ranking_weight"] == ranking_weight]
        aggregates.append(
            {
                "ranking_weight": ranking_weight,
                "mean_validation_error_auprc": float(
                    np.mean([run["validation"]["error_auprc"] for run in selected])
                ),
                "std_validation_error_auprc": float(
                    np.std([run["validation"]["error_auprc"] for run in selected], ddof=1)
                ),
                "mean_validation_auroc": float(
                    np.mean([run["validation"]["correctness_auroc"] for run in selected])
                ),
                "mean_best_epoch": float(np.mean([run["best_epoch"] for run in selected])),
            }
        )
    winner = max(
        aggregates,
        key=lambda item: (item["mean_validation_error_auprc"], item["mean_validation_auroc"]),
    )
    winning_weight = winner["ranking_weight"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    test_runs = []
    for seed in (1, 2, 3):
        run = next(
            item for item in runs if item["ranking_weight"] == winning_weight and item["seed"] == seed
        )
        head = QHead(train["generated_features"].shape[1], hidden_size=512).to(device)
        head.load_state_dict(states[(winning_weight, seed)])
        test_metrics = evaluate_head(
            head, run["temperature"], test["generated_features"], test["labels"], device
        )
        test_runs.append({"seed": seed, "best_epoch": run["best_epoch"], **test_metrics})
        torch.save(
            states[(winning_weight, seed)],
            args.output.with_name(f"early_stop_verifier_weight_{winning_weight:g}_seed_{seed}.pt"),
        )

    metric_names = (
        "correctness_auroc",
        "error_auprc",
        "brier",
        "ece_10_bin",
        "pearson_correctness",
    )
    test_summary = {
        name: {
            "mean": float(np.mean([run[name] for run in test_runs])),
            "std": float(np.std([run[name] for run in test_runs], ddof=1)),
        }
        for name in metric_names
    }
    result = {
        "fixed_config": {"architecture": "mlp_512", "learning_rate": 1e-3},
        "selection": "mean validation error_auprc across 3 seeds, then mean validation auroc",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "weight_aggregates": aggregates,
        "runs": runs,
        "winning_ranking_weight": winning_weight,
        "test_runs": test_runs,
        "test_summary": test_summary,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner, "test_summary": test_summary}, indent=2))


if __name__ == "__main__":
    main()
