"""Small architecture/hyperparameter sweep over cached Qwen3-VL features."""

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.approach1_latent_self_correction.train_q_heads import (
    QHead,
    evaluate_head,
    fit_temperature,
    metrics,
    ranking_loss,
)


class LinearProbe(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.output = nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features.float()).squeeze(-1)


def make_head(architecture: str, input_size: int) -> nn.Module:
    if architecture == "linear":
        return LinearProbe(input_size)
    return QHead(input_size, hidden_size=int(architecture.removeprefix("mlp_")))


def train_one(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    device: torch.device,
    architecture: str,
    learning_rate: float,
    ranking_weight: float,
    seed: int,
    epochs: int,
    pair_reference: torch.Tensor | None = None,
) -> tuple[nn.Module, float, dict]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = make_head(architecture, train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        permutation = torch.randperm(len(train_labels), generator=generator)
        head.train()
        for start in range(0, len(permutation), 512):
            indices = permutation[start : start + 512]
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            logits = head(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + ranking_weight * ranking_loss(logits, labels)
            if pair_reference is not None:
                wrong = labels == 0
                if wrong.any():
                    references = pair_reference[indices].to(device)
                    reference_logits = head(references[wrong])
                    loss = loss + ranking_weight * F.softplus(
                        -(reference_logits - logits[wrong])
                    ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    head.eval()
    with torch.inference_mode():
        validation_logits = head(validation_features.to(device)).cpu().numpy()
    temperature = fit_temperature(validation_logits, validation_labels.numpy())
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(validation_logits / temperature, -30, 30))
    )
    return head, temperature, metrics(validation_labels.numpy(), probabilities)


def sweep_approach(
    name: str,
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_labels: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    pair_reference: torch.Tensor | None = None,
) -> tuple[list[dict], dict, dict, dict]:
    runs = []
    best_key = (-1.0, -1.0)
    best_config = None
    best_state = None
    best_temperature = None
    configurations = itertools.product(
        ("linear", "mlp_256", "mlp_512"),
        (3e-4, 1e-3),
        (0.0, 0.25),
        (1, 2, 3),
    )
    for run_number, (architecture, learning_rate, ranking_weight, seed) in enumerate(
        configurations, start=1
    ):
        config = {
            "architecture": architecture,
            "learning_rate": learning_rate,
            "ranking_weight": ranking_weight,
            "seed": seed,
        }
        head, temperature, validation_metrics = train_one(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            device,
            architecture,
            learning_rate,
            ranking_weight,
            seed,
            epochs,
            pair_reference,
        )
        run = {"config": config, "temperature": temperature, "validation": validation_metrics}
        runs.append(run)
        key = (validation_metrics["error_auprc"], validation_metrics["correctness_auroc"])
        if key > best_key:
            best_key = key
            best_config = config
            best_temperature = temperature
            best_state = copy.deepcopy(head.state_dict())
        print(
            f"[{name} {run_number}/36] {config} "
            f"error_auprc={validation_metrics['error_auprc']:.4f} "
            f"auroc={validation_metrics['correctness_auroc']:.4f}",
            flush=True,
        )
        del head

    assert best_config is not None and best_state is not None and best_temperature is not None
    best_head = make_head(best_config["architecture"], train_features.shape[1]).to(device)
    best_head.load_state_dict(best_state)
    test_metrics = evaluate_head(best_head, best_temperature, test_features, test_labels, device)
    return runs, best_config, best_state, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train = torch.load(args.train, map_location="cpu", weights_only=True)
    validation = torch.load(args.validation, map_location="cpu", weights_only=True)
    test = torch.load(args.test, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    states = {}

    for name, feature_key, reference_key in (
        ("pre_generation", "prompt_features", None),
        ("candidate_verifier", "generated_features", "reference_features"),
    ):
        runs, best_config, best_state, test_metrics = sweep_approach(
            name,
            train[feature_key],
            validation[feature_key],
            test[feature_key],
            train["labels"],
            validation["labels"],
            test["labels"],
            device,
            args.epochs,
            train[reference_key] if reference_key else None,
        )
        results[name] = {
            "selection_metric": "validation error_auprc, then correctness_auroc",
            "best_config": best_config,
            "best_validation": next(run["validation"] for run in runs if run["config"] == best_config),
            "test": test_metrics,
            "validation_runs": runs,
        }
        states[name] = best_state

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, state in states.items():
        torch.save(state, args.output.with_name(f"sweep_best_{name}.pt"))
    print(json.dumps({name: {"best_config": value["best_config"], "test": value["test"]} for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
