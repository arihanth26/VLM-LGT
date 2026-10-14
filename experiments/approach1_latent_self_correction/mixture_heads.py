# mixture_heads.py
#
# Training and evaluation of correctness heads on a dataset mixture, for the
# old loss and the new loss. Three head kinds share one code path:
#   pre        separate pre-generation head, reads the prompt state
#   verifier   separate answer-conditioned head, reads the prompt-plus-answer state
#   unified    one head that reads both states, in joint and pre-generation modes
#
# Loss terms (old loss is ground_weight = 0, new loss is ground_weight > 0):
#   bce      correctness cross-entropy on the clean input, P(S_confidence | h_answer, h_question)
#   pre      unified only: the same cross-entropy with the answer half zeroed
#   rank     batch ranking of correct above wrong
#   pair     L_contrastive(h_predicted, h_correct): the reference answer must score above a
#            wrong predicted answer for the same question and image
#   ground   L_contrastive(h_clean, h_corrupt): the clean input must score above the same
#            input with a corrupted image, so confidence tracks visual grounding
#
# Evaluation reports pooled and per-dataset metrics, and how far scores drop when the
# image is corrupted (the grounding metrics).

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from experiments.approach1_latent_self_correction.sweep_q_heads import make_head
from experiments.approach1_latent_self_correction.train_q_heads import (
    fit_temperature,
    metrics,
    ranking_loss,
)
from experiments.approach1_latent_self_correction.unified_q_head import UnifiedQHead, to_probabilities

KEY_METRICS = ("correctness_auroc", "error_auprc", "brier", "ece_10_bin", "pearson_correctness")
MIN_DATASET_ROWS = 30


@dataclass
class HeadSpec:
    """One head configuration: its kind, architecture, optimizer, loss weights, and seed."""

    kind: str
    architecture: str
    learning_rate: float
    rank_weight: float = 0.0
    pair_weight: float = 0.0
    pre_weight: float = 0.0
    ground_weight: float = 0.0
    ground_margin: float = 0.0
    ground_on: str = "correct"
    seed: int = 1

    @property
    def family(self) -> str:
        """Whether this configuration uses the old loss or the new loss."""
        return "new" if self.ground_weight > 0 else "old"

    def config(self) -> dict:
        """The configuration without the seed, for grouping runs."""
        values = asdict(self)
        values.pop("seed")
        return values


def modes_for(kind: str) -> tuple[str, ...]:
    """Names of the scoring modes a head kind produces."""
    return ("joint", "pre_generation") if kind == "unified" else ("main",)


def main_mode(kind: str) -> str:
    """The mode that carries the primary correctness loss."""
    return "joint" if kind == "unified" else "main"


def build_head(spec: HeadSpec, feature_size: int) -> torch.nn.Module:
    """Create the network for a head spec."""
    if spec.kind == "unified":
        return UnifiedQHead(feature_size, spec.architecture)
    return make_head(spec.architecture, feature_size)


def head_logits(head, kind: str, prompt, generated, modes: tuple[str, ...]) -> dict:
    """Run a head on one set of inputs and return a logit tensor per requested mode."""
    if kind == "pre":
        return {"main": head(prompt)}
    if kind == "verifier":
        return {"main": head(generated)}
    result = {}
    if "joint" in modes:
        result["joint"] = head(prompt, generated)
    if "pre_generation" in modes:
        result["pre_generation"] = head(prompt, None)
    return result


def head_loss(head, spec: HeadSpec, batch: dict) -> tuple[torch.Tensor, dict]:
    """Combine the active loss terms for one batch and return the total with its parts."""
    labels = batch["labels"]
    kind = spec.kind
    needed = modes_for(kind) if (kind == "unified" and spec.pre_weight) else (main_mode(kind),)
    clean = head_logits(head, kind, batch["prompt"], batch["generated"], needed)
    primary = clean[main_mode(kind)]
    parts = {"bce": F.binary_cross_entropy_with_logits(primary, labels)}
    if kind == "unified" and spec.pre_weight:
        parts["pre"] = spec.pre_weight * F.binary_cross_entropy_with_logits(clean["pre_generation"], labels)
    if spec.rank_weight:
        parts["rank"] = spec.rank_weight * ranking_loss(primary, labels)
    if spec.pair_weight and kind != "pre" and "reference" in batch:
        wrong = labels == 0
        if wrong.any():
            if kind == "verifier":
                reference_logits = head(batch["reference"][wrong])
            else:
                reference_logits = head(batch["prompt"][wrong], batch["reference"][wrong])
            parts["pair"] = spec.pair_weight * F.softplus(-(reference_logits - primary[wrong])).mean()
    if spec.ground_weight:
        corrupt = head_logits(head, kind, batch["prompt_corrupt"], batch["generated_corrupt"], needed)
        mask = labels == 1 if spec.ground_on == "correct" else torch.ones_like(labels, dtype=torch.bool)
        if mask.any():
            terms = [F.softplus(spec.ground_margin - (clean[m] - corrupt[m]))[mask].mean() for m in clean]
            parts["ground"] = spec.ground_weight * sum(terms) / len(terms)
    return sum(parts.values()), {name: float(value.detach()) for name, value in parts.items()}


def to_device(data: dict, device: torch.device) -> dict:
    """Move the tensors a head needs onto the device, under the names the loss expects."""
    mapping = {
        "prompt": "prompt_features",
        "generated": "generated_features",
        "prompt_corrupt": "prompt_features_corrupt",
        "generated_corrupt": "generated_features_corrupt",
        "reference": "reference_features",
        "labels": "labels",
    }
    return {name: data[key].to(device) for name, key in mapping.items() if key in data}


def predict_logits(head, kind: str, tensors: dict, corrupt: bool, device: torch.device, batch_size: int = 4096) -> dict:
    """Score a whole split in batches and return logits per mode as numpy arrays."""
    head.eval()
    prompt = tensors["prompt_corrupt" if corrupt else "prompt"]
    generated = tensors["generated_corrupt" if corrupt else "generated"]
    modes = modes_for(kind)
    chunks = {mode: [] for mode in modes}
    with torch.inference_mode():
        for start in range(0, len(prompt), batch_size):
            outputs = head_logits(
                head, kind, prompt[start : start + batch_size].float(), generated[start : start + batch_size].float(), modes
            )
            for mode in modes:
                chunks[mode].append(outputs[mode].float().cpu())
    return {mode: torch.cat(chunks[mode]).numpy() for mode in modes}


def key_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """The headline metrics for one set of probabilities."""
    values = metrics(labels, probabilities)
    return {name: float(values[name]) for name in KEY_METRICS}


def grounding_metrics(clean: np.ndarray, corrupt: np.ndarray, labels: np.ndarray) -> dict:
    """How often and by how much a score falls when the image is corrupted."""
    result = {}
    for name, mask in (("correct", labels == 1), ("all", np.ones_like(labels, dtype=bool))):
        valid = mask & ~np.isnan(clean) & ~np.isnan(corrupt)
        drop = clean[valid] - corrupt[valid]
        result[name] = {
            "mean_drop": float(drop.mean()) if len(drop) else float("nan"),
            "order_accuracy": float(((drop > 0).mean() + 0.5 * (drop == 0).mean())) if len(drop) else float("nan"),
            "count": int(valid.sum()),
        }
    return result


def evaluate_split(
    head, spec: HeadSpec, tensors: dict, labels: np.ndarray, dataset_id: np.ndarray, names: list[str],
    temperatures: dict, device: torch.device, extra: dict | None = None,
) -> dict:
    """Pooled, per-dataset, and grounding metrics for a trained head on one split.

    extra maps a corruption name to its corrupted test features, so grounding learned under one
    corruption can also be measured against others.
    """
    clean_logits = predict_logits(head, spec.kind, tensors, False, device)
    corrupt_logits = predict_logits(head, spec.kind, tensors, True, device)
    pooled, per_dataset, grounding, probabilities = {}, {}, {}, {}
    for mode in modes_for(spec.kind):
        clean = to_probabilities(clean_logits[mode], temperatures[mode])
        corrupt = to_probabilities(corrupt_logits[mode], temperatures[mode])
        probabilities[mode] = {"clean": clean, "corrupt": corrupt}
        pooled[mode] = key_metrics(labels, clean)
        grounding[mode] = {"pooled": grounding_metrics(clean, corrupt, labels)}
        for index, name in enumerate(names):
            rows = dataset_id == index
            if rows.sum() >= MIN_DATASET_ROWS and 0 < labels[rows].mean() < 1:
                per_dataset.setdefault(name, {})[mode] = key_metrics(labels[rows], clean[rows])
                grounding[mode][name] = grounding_metrics(clean[rows], corrupt[rows], labels[rows])
    result = {"pooled": pooled, "per_dataset": per_dataset, "grounding": grounding, "probabilities": probabilities}
    if extra:
        result["grounding_extra"] = extra_grounding(head, spec, tensors, labels, dataset_id, names, temperatures, device, extra, probabilities)
    return result


def extra_grounding(head, spec, tensors, labels, dataset_id, names, temperatures, device, extra, probabilities) -> dict:
    """Grounding metrics of the same clean scores against each extra corruption of the test split."""
    result = {}
    for corruption, features in extra.items():
        swapped = {**tensors, "prompt_corrupt": features["prompt_corrupt"], "generated_corrupt": features["generated_corrupt"]}
        logits = predict_logits(head, spec.kind, swapped, True, device)
        for mode in modes_for(spec.kind):
            clean = probabilities[mode]["clean"]
            other = to_probabilities(logits[mode], temperatures[mode])
            entry = {"pooled": grounding_metrics(clean, other, labels)}
            for index, name in enumerate(names):
                rows = dataset_id == index
                if rows.sum() >= MIN_DATASET_ROWS and 0 < labels[rows].mean() < 1:
                    entry[name] = grounding_metrics(clean[rows], other[rows], labels[rows])
            result.setdefault(corruption, {})[mode] = entry
    return result


def selection_key(pooled: dict, kind: str) -> tuple[float, float]:
    """Rank a run by validation error AUPRC then AUROC, averaged over its modes."""
    modes = modes_for(kind)
    return (
        float(np.mean([pooled[m]["error_auprc"] for m in modes])),
        float(np.mean([pooled[m]["correctness_auroc"] for m in modes])),
    )


def train_head(
    spec: HeadSpec, train: dict, validation: dict, test: dict, names: list[str], device: torch.device,
    max_epochs: int = 40, patience: int = 4, batch_size: int = 1024, extra_tests: dict | None = None,
) -> dict:
    """Train one head with early stopping on validation and evaluate the best state on validation and test."""
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    feature_size = train["prompt"].shape[1]
    head = build_head(spec, feature_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=spec.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(spec.seed)
    val_labels = validation["labels"].cpu().numpy()
    val_ids = validation["dataset_id"].cpu().numpy()

    best, best_key, stale = None, (-1.0, -1.0), 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.randperm(len(train["labels"]), generator=generator).to(device)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch = {name: tensor[rows].float() if name != "labels" else tensor[rows] for name, tensor in train.items() if name != "dataset_id"}
            loss, _ = head_loss(head, spec, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logits = predict_logits(head, spec.kind, validation, False, device)
        temperatures = {m: fit_temperature(logits[m], val_labels) for m in logits}
        pooled = {m: key_metrics(val_labels, to_probabilities(logits[m], temperatures[m])) for m in logits}
        key = selection_key(pooled, spec.kind)
        if key > best_key:
            best_key, stale = key, 0
            best = {"epoch": epoch, "temperatures": temperatures, "val_pooled": pooled, "state": copy.deepcopy(head.state_dict())}
        else:
            stale += 1
            if stale >= patience:
                break
    assert best is not None
    head.load_state_dict(best["state"])
    test_eval = evaluate_split(
        head, spec, test, test["labels"].cpu().numpy(), test["dataset_id"].cpu().numpy(), names, best["temperatures"], device,
        extra_tests,
    )
    return {
        "spec": asdict(spec),
        "epoch": best["epoch"],
        "val_pooled": best["val_pooled"],
        "temperatures": best["temperatures"],
        "test": test_eval,
    }
