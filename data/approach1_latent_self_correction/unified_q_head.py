# unified_q_head.py
#
# One correctness head that reads the pre-generation latent and the
# answer-conditioned (verifier) latent together. The two cached Qwen3-VL
# states are layer-normalized separately, concatenated, and sent through a
# single network that outputs one correctness logit.
#
# The same weights also work before an answer exists. In that mode the
# answer half of the input is zeroed and a flag bit says so, which gives the
# "should we correct?" gate a pre-generation score without a second model.
#
# Loss, for a batch with prompt latents p, answer latents g, reference-answer
# latents r, and correctness labels y:
#   L = BCE(f(p, g), y)                          joint mode, the verifier
#     + pre_weight  * BCE(f(p, none), y)         pre-generation mode, same weights
#     + rank_weight * rank(f(p, g), y)           correct above wrong across the batch
#     + pair_weight * softplus(f(p, g) - f(p, r)) on wrong rows, reference above candidate
# The pre term is what keeps the head from leaning only on the answer half.
# Without it, the answer latent is the easier signal and the prompt half of the
# weights gets little gradient, so pre-generation mode would be poor.

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.approach1_latent_self_correction.train_q_heads import (
    fit_temperature,
    metrics,
    ranking_loss,
)

MODES = ("joint", "pre_generation")


@dataclass
class UnifiedConfig:
    """Hyperparameters for one unified head run."""

    architecture: str = "mlp_512"
    learning_rate: float = 1e-3
    pre_weight: float = 1.0
    rank_weight: float = 0.0
    pair_weight: float = 0.0
    seed: int = 1


class UnifiedQHead(nn.Module):
    """Correctness head over concatenated prompt and answer latents."""

    def __init__(self, input_size: int, architecture: str = "mlp_512", dropout: float = 0.1):
        super().__init__()
        self.prompt_norm = nn.LayerNorm(input_size)
        self.answer_norm = nn.LayerNorm(input_size)
        fused_size = 2 * input_size + 1
        if architecture == "linear":
            self.body = nn.Linear(fused_size, 1)
        else:
            hidden = int(architecture.removeprefix("mlp_"))
            self.body = nn.Sequential(
                nn.Linear(fused_size, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, prompt: torch.Tensor, answer: torch.Tensor | None = None) -> torch.Tensor:
        """Score correctness from the prompt latent, plus the answer latent when given."""
        prompt = self.prompt_norm(prompt.float())
        if answer is None:
            answer = torch.zeros_like(prompt)
            flag = prompt.new_zeros(prompt.shape[0], 1)
        else:
            answer = self.answer_norm(answer.float())
            flag = prompt.new_ones(prompt.shape[0], 1)
        return self.body(torch.cat([prompt, answer, flag], dim=-1)).squeeze(-1)


def unified_loss(
    head: UnifiedQHead,
    prompt: torch.Tensor,
    answer: torch.Tensor,
    labels: torch.Tensor,
    config: UnifiedConfig,
    reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Combine the joint, pre-generation, ranking, and pairwise terms into one loss."""
    joint_logits = head(prompt, answer)
    parts = {"joint": F.binary_cross_entropy_with_logits(joint_logits, labels)}
    if config.pre_weight:
        pre_logits = head(prompt, None)
        parts["pre"] = config.pre_weight * F.binary_cross_entropy_with_logits(pre_logits, labels)
    if config.rank_weight:
        parts["rank"] = config.rank_weight * ranking_loss(joint_logits, labels)
    if config.pair_weight and reference is not None:
        wrong = labels == 0
        if wrong.any():
            reference_logits = head(prompt[wrong], reference[wrong])
            parts["pair"] = config.pair_weight * F.softplus(
                -(reference_logits - joint_logits[wrong])
            ).mean()
    return sum(parts.values()), {name: float(value.detach()) for name, value in parts.items()}


def predict_logits(
    head: UnifiedQHead,
    prompt: torch.Tensor,
    answer: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run the head over cached features in batches and return raw logits."""
    head.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(prompt), batch_size):
            batch_prompt = prompt[start : start + batch_size].to(device)
            batch_answer = None if answer is None else answer[start : start + batch_size].to(device)
            outputs.append(head(batch_prompt, batch_answer).cpu())
    return torch.cat(outputs).numpy()


def mode_inputs(data: dict, mode: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pick the (prompt, answer) feature pair that a given mode reads."""
    if mode == "joint":
        return data["prompt_features"], data["generated_features"]
    return data["prompt_features"], None


def to_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits to probabilities after dividing by a calibration temperature."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30, 30)))


def evaluate_modes(
    head: UnifiedQHead,
    temperatures: dict[str, float],
    data: dict,
    device: torch.device,
) -> dict[str, dict]:
    """Compute calibrated metrics for the joint and pre-generation modes on one split."""
    labels = data["labels"].numpy()
    results = {}
    for mode in MODES:
        prompt, answer = mode_inputs(data, mode)
        probabilities = to_probabilities(predict_logits(head, prompt, answer, device), temperatures[mode])
        results[mode] = {"temperature": temperatures[mode], **metrics(labels, probabilities)}
    return results


def fit_mode_temperatures(head: UnifiedQHead, data: dict, device: torch.device) -> dict[str, float]:
    """Fit one calibration temperature per mode on validation features."""
    labels = data["labels"].numpy()
    temperatures = {}
    for mode in MODES:
        prompt, answer = mode_inputs(data, mode)
        temperatures[mode] = fit_temperature(predict_logits(head, prompt, answer, device), labels)
    return temperatures


def selection_score(results: dict[str, dict], select: str) -> tuple[float, float]:
    """Rank a run by validation error AUPRC then AUROC, over one mode or the mean of both."""
    modes = MODES if select == "mean" else (select,)
    return (
        float(np.mean([results[mode]["error_auprc"] for mode in modes])),
        float(np.mean([results[mode]["correctness_auroc"] for mode in modes])),
    )


def train_unified(
    train: dict,
    validation: dict,
    config: UnifiedConfig,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 5,
    select: str = "mean",
    batch_size: int = 512,
) -> dict:
    """Train one unified head with early stopping and return its best state and validation results."""
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    head = UnifiedQHead(train["prompt_features"].shape[1], config.architecture).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(config.seed)
    use_reference = bool(config.pair_weight) and "reference_features" in train

    best = None
    best_key = (-1.0, -1.0)
    stale = 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.randperm(len(train["labels"]), generator=generator)
        losses = []
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            reference = train["reference_features"][rows].to(device) if use_reference else None
            loss, _ = unified_loss(
                head,
                train["prompt_features"][rows].to(device),
                train["generated_features"][rows].to(device),
                train["labels"][rows].to(device),
                config,
                reference,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        temperatures = fit_mode_temperatures(head, validation, device)
        results = evaluate_modes(head, temperatures, validation, device)
        key = selection_score(results, select)
        if key > best_key:
            best_key = key
            stale = 0
            best = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "temperatures": temperatures,
                "validation": results,
                "state": copy.deepcopy(head.state_dict()),
            }
        else:
            stale += 1
            if stale >= patience:
                break
    assert best is not None
    best["epochs_run"] = epoch
    return best
