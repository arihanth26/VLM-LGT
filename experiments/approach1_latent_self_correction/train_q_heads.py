# train_q_heads.py
#
# Trains and compares a pre-generation correctness head and an
# answer-conditioned pairwise verifier on frozen Qwen3-VL features.

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QHead(nn.Module):
    """Small correctness MLP over one frozen dense representation."""

    def __init__(self, input_size: int, hidden_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.float()).squeeze(-1)


def ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Rank every correct item in a batch above every incorrect item."""
    positive = logits[labels == 1]
    negative = logits[labels == 0]
    if not len(positive) or not len(negative):
        return logits.new_zeros(())
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Choose a scalar temperature that minimizes validation BCE."""
    temperatures = np.geomspace(0.1, 10.0, 300)
    losses = []
    for temperature in temperatures:
        scaled = np.clip(logits / temperature, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-scaled))
        losses.append(
            -np.mean(
                labels * np.log(probabilities + 1e-9)
                + (1 - labels) * np.log(1 - probabilities + 1e-9)
            )
        )
    return float(temperatures[int(np.argmin(losses))])


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC AUC through pairwise score comparisons."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean()))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute average precision for a binary target sorted by descending score."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    positives = sorted_labels.sum()
    if positives == 0:
        return 0.0
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Calculate discrimination, calibration, and selective-risk metrics."""
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for bin_index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (probabilities >= lower) & (
            (probabilities <= upper) if bin_index == len(bins) - 2 else (probabilities < upper)
        )
        if mask.any():
            ece += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    order = np.argsort(-probabilities)
    coverage = {}
    for fraction in (1.0, 0.9, 0.8, 0.5):
        count = max(1, math.ceil(len(labels) * fraction))
        coverage[str(fraction)] = float(labels[order[:count]].mean())
    return {
        "correctness_auroc": roc_auc(labels, probabilities),
        "error_auprc": average_precision(1 - labels, 1 - probabilities),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece_10_bin": float(ece),
        "pearson_correctness": float(np.corrcoef(probabilities, labels)[0, 1]),
        "accuracy_at_coverage": coverage,
    }


def train_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    pair_reference: torch.Tensor | None = None,
) -> tuple[QHead, float]:
    """Train one Q head with BCE, global ranking, and optional same-example ranking."""
    head = QHead(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(42)
    for epoch in range(epochs):
        permutation = torch.randperm(len(train_labels), generator=generator)
        head.train()
        losses = []
        for start in range(0, len(permutation), 512):
            indices = permutation[start : start + 512]
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            logits = head(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + 0.25 * ranking_loss(logits, labels)
            if pair_reference is not None:
                wrong = labels == 0
                if wrong.any():
                    reference_features = pair_reference[indices].to(device)
                    reference_logits = head(reference_features[wrong])
                    loss = loss + 0.25 * F.softplus(
                        -(reference_logits - logits[wrong])
                    ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f}")

    head.eval()
    with torch.inference_mode():
        validation_logits = head(validation_features.to(device)).cpu().numpy()
    temperature = fit_temperature(validation_logits, validation_labels.numpy())
    return head, temperature


def evaluate_head(
    head: QHead,
    temperature: float,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict:
    """Evaluate a calibrated correctness head on frozen features."""
    head.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(labels), 1024):
            logits.append(head(features[start : start + 1024].to(device)).cpu())
    values = torch.cat(logits).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(values / temperature, -30, 30)))
    return {"temperature": temperature, **metrics(labels.numpy(), probabilities)}


def main() -> None:
    """Train both heads and write their test-set comparison."""
    parser = argparse.ArgumentParser(description="Train pre-generation and verifier Q heads.")
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

    pre_head, pre_temperature = train_head(
        train["prompt_features"],
        train["labels"],
        validation["prompt_features"],
        validation["labels"],
        device,
        args.epochs,
    )
    verifier, verifier_temperature = train_head(
        train["generated_features"],
        train["labels"],
        validation["generated_features"],
        validation["labels"],
        device,
        args.epochs,
        pair_reference=train["reference_features"],
    )
    results = {
        "pre_generation": evaluate_head(
            pre_head,
            pre_temperature,
            test["prompt_features"],
            test["labels"],
            device,
        ),
        "candidate_verifier": evaluate_head(
            verifier,
            verifier_temperature,
            test["generated_features"],
            test["labels"],
            device,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    torch.save(pre_head.state_dict(), args.output.with_name("pre_generation_q_head.pt"))
    torch.save(verifier.state_dict(), args.output.with_name("candidate_verifier_q_head.pt"))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
