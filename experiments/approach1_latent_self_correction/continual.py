# continual.py
#
# Sequential (continual) training of correctness heads across datasets. A head is
# trained on dataset A, then on B, then on C, and after every stage it is scored on
# the test split of every dataset, which gives a matrix of how well each dataset is
# remembered. This answers whether training on later datasets makes the head forget
# earlier ones, and whether the new visual-grounding loss or a small replay buffer
# changes that.
#
# Metrics, for an order of datasets and a score R[stage][dataset] (AUROC):
#   final_avg   mean of R[last][d] over all datasets
#   forgetting  for each dataset trained before the last stage, the best score it had
#               from the stage it was trained to the stage before the last, minus its
#               final score, averaged
#   bwt         backward transfer: final score minus the score right after its own stage,
#               averaged over datasets trained before the last stage
#   retention   final score minus the score of a head trained on that dataset alone
#
# Built on the cached features and the losses of mixture_heads.py.

import copy

import numpy as np
import torch

from experiments.approach1_latent_self_correction.mixture_heads import (
    HeadSpec,
    build_head,
    grounding_metrics,
    head_loss,
    key_metrics,
    modes_for,
    predict_logits,
    selection_key,
)
from experiments.approach1_latent_self_correction.train_q_heads import fit_temperature
from experiments.approach1_latent_self_correction.unified_q_head import to_probabilities


def take(tensors: dict, rows: torch.Tensor) -> dict:
    """Select rows from every tensor in a split."""
    return {name: tensor[rows] for name, tensor in tensors.items()}


def concat(parts: list[dict]) -> dict:
    """Concatenate splits along the row axis, keeping the tensors they all share."""
    keys = set.intersection(*(set(part) for part in parts))
    return {key: torch.cat([part[key] for part in parts]) for key in keys}


def split_by_dataset(tensors: dict, count: int) -> list[dict]:
    """Split a pooled split into one split per dataset, by its dataset id."""
    ids = tensors["dataset_id"]
    return [take(tensors, torch.where(ids == index)[0]) for index in range(count)]


def replay_sample(parts: list[dict], fraction: float, seed: int) -> dict | None:
    """A fixed random fraction of every earlier dataset's training rows, or None for no replay."""
    if not parts or fraction <= 0:
        return None
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for part in parts:
        count = len(part["labels"])
        keep = torch.randperm(count, generator=generator)[: max(1, int(round(fraction * count)))].to(part["labels"].device)
        chosen.append(take(part, keep))
    return concat(chosen)


def fit_stage(
    head, spec: HeadSpec, train: dict, validation: dict, device: torch.device,
    max_epochs: int, patience: int, batch_size: int, seed: int,
) -> dict:
    """Continue training an existing head on one stage, with early stopping on that stage's validation."""
    optimizer = torch.optim.AdamW(head.parameters(), lr=spec.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    labels = validation["labels"].cpu().numpy()
    best, best_key, stale = None, (-1.0, -1.0), 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.randperm(len(train["labels"]), generator=generator).to(device)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch = {name: tensor[rows] if name == "labels" else tensor[rows].float() for name, tensor in train.items() if name != "dataset_id"}
            loss, _ = head_loss(head, spec, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logits = predict_logits(head, spec.kind, validation, False, device)
        temperatures = {m: fit_temperature(logits[m], labels) for m in logits}
        pooled = {m: key_metrics(labels, to_probabilities(logits[m], temperatures[m])) for m in logits}
        key = selection_key(pooled, spec.kind)
        if key > best_key:
            best_key, stale = key, 0
            best = {"temperatures": temperatures, "state": copy.deepcopy(head.state_dict()), "epoch": epoch}
        else:
            stale += 1
            if stale >= patience:
                break
    head.load_state_dict(best["state"])
    return {"temperatures": best["temperatures"], "epoch": best["epoch"]}


def evaluate_dataset(head, spec: HeadSpec, test: dict, temperatures: dict, device: torch.device) -> dict:
    """Ranking quality and grounding of a head on one dataset's test split, per mode."""
    labels = test["labels"].cpu().numpy()
    clean = predict_logits(head, spec.kind, test, False, device)
    corrupt = predict_logits(head, spec.kind, test, True, device)
    result = {}
    for mode in modes_for(spec.kind):
        p_clean = to_probabilities(clean[mode], temperatures[mode])
        p_corrupt = to_probabilities(corrupt[mode], temperatures[mode])
        scores = key_metrics(labels, p_clean)
        ground = grounding_metrics(p_clean, p_corrupt, labels)["correct"]
        result[mode] = {
            "auroc": scores["correctness_auroc"],
            "error_auprc": scores["error_auprc"],
            "order_accuracy": ground["order_accuracy"],
        }
    return result


def run_sequence(
    spec: HeadSpec, order: list[int], train: list[dict], validation: list[dict], test: list[dict],
    device: torch.device, replay: float = 0.0, max_epochs: int = 40, patience: int = 4, batch_size: int = 1024,
) -> dict:
    """Train one head through the datasets in the given order and score every dataset after each stage."""
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    feature_size = train[0]["prompt"].shape[1]
    head = build_head(spec, feature_size).to(device)
    stages = []
    for position, dataset in enumerate(order):
        buffer = replay_sample([train[earlier] for earlier in order[:position]], replay, spec.seed + position)
        stage_train = train[dataset] if buffer is None else concat([train[dataset], buffer])
        info = fit_stage(head, spec, stage_train, validation[dataset], device, max_epochs, patience, batch_size, spec.seed * 100 + position)
        scores = {index: evaluate_dataset(head, spec, test[index], info["temperatures"], device) for index in range(len(test))}
        stages.append({"trained_on": dataset, "epoch": info["epoch"], "scores": scores})
    return {"order": order, "stages": stages}


def train_isolated(spec: HeadSpec, train: dict, validation: dict, test: list[dict], device: torch.device,
                   max_epochs: int = 40, patience: int = 4, batch_size: int = 1024) -> dict:
    """Train a fresh head on one dataset alone and score it on every dataset's test split."""
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    head = build_head(spec, train["prompt"].shape[1]).to(device)
    info = fit_stage(head, spec, train, validation, device, max_epochs, patience, batch_size, spec.seed * 100)
    return {index: evaluate_dataset(head, spec, test[index], info["temperatures"], device) for index in range(len(test))}


def matrix_of(sequence: dict, mode: str, field: str = "auroc") -> np.ndarray:
    """The stage-by-dataset matrix of one field, with rows in stage order and columns in dataset id order."""
    count = len(sequence["stages"][0]["scores"])
    return np.array([[stage["scores"][d][mode][field] for d in range(count)] for stage in sequence["stages"]])


def sequence_metrics(matrix: np.ndarray, order: list[int], isolated: np.ndarray | None = None) -> dict:
    """Final average, forgetting, backward transfer, and retention for one sequence matrix."""
    last = len(order) - 1
    final = matrix[last]
    earlier = order[:last]
    forgetting, bwt = [], []
    for position, dataset in enumerate(earlier):
        forgetting.append(matrix[position:last, dataset].max() - final[dataset])
        bwt.append(final[dataset] - matrix[position, dataset])
    result = {
        "final_avg": float(final.mean()),
        "forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "bwt": float(np.mean(bwt)) if bwt else 0.0,
    }
    if isolated is not None:
        result["retention"] = float((final - isolated).mean())
    return result
