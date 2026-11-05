# sweep_pre_head.py
#
# Sweeps how the pre-generation head is trained, with its input held fixed. The input is the
# cached question embedding (the prompt state in q_features/<dataset>_<split>_2b.pt), the same
# tensor for every target, loss and configuration below; only the training changes.
#
# Why change the training: the current pre-generation head learns whether one greedy answer was
# right. That label is noisy for a question-level score (a question the model gets right 6 times
# in 10 is labelled 1 or 0 by chance), and it is not what the score is used for, which is
# deciding which questions deserve more samples. Targets swept:
#   greedy        greedy answer correct (the current target)
#   pass_rate     fraction of K samples at the label temperature that are correct (soft target)
#   blend         mean of the two
#   any_correct   at least one of the K samples correct (the question is solvable by sampling)
# plus an optional pairwise ranking term on the pass rate, the architecture, learning rate,
# weight decay and dropout. Every configuration is trained on three seeds with early stopping.
#
# Selection is on validation by two criteria, both reported on test:
#   gate      AURC of the greedy answers ordered by the score (lower is better)
#   routing   mean accuracy gain when the lowest-scored questions are sent to the chosen
#             candidate search (results/candidate_search.json) and the rest keep greedy
# Test also reports AUROC against greedy correctness for comparison with earlier results, the
# rank correlation with the pass rate, and the exported pre-generation heads as references.
#
# Pass-rate labels come from candidate_pools.py: --labels-only pools on train, and the first K
# samples of the scored pools on val and test, all at the label temperature.

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.export_heads import load_head, score
from experiments.approach1_latent_self_correction.selection_metrics import aurc, compute_curve
from experiments.approach1_latent_self_correction.train_q_heads import fit_temperature, roc_auc
from experiments.approach1_latent_self_correction.unified_q_head import to_probabilities

HERE = Path(__file__).parent
FEATURE_DIR = HERE / "q_features"
POOL_DIR = HERE / "results" / "pools"
RESULTS_DIR = HERE / "results"
HEADS = HERE.parents[1] / "trained_heads" / "approach1" / "winners" / "mixture"
REFERENCE_HEADS = {"exported pre-generation (MLP)": "pre_new_seed1.pt", "exported pre-generation (linear)": "pre_new_linear_seed1.pt",
                   "exported unified, pre-generation mode": "unified_new_seed1.pt"}
TARGETS = ("greedy", "pass_rate", "blend", "any_correct")
SEEDS = (1, 2, 3)


def load_pass_rates(temperature: float, split: str, samples: int) -> dict[tuple[str, int], float]:
    """Fraction of the first K samples that are correct, per question, from the pools at one temperature."""
    rates = {}
    for path in sorted((POOL_DIR / f"T{temperature:g}" / split).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "samples" in record and len(record["samples"]) >= samples:
                    rates[(record["dataset"], record["index"])] = float(np.mean([c["correct"] for c in record["samples"][:samples]]))
    return rates


def load_pools(split: str, temperature: float) -> dict[tuple[str, int], dict]:
    """Scored pools of one split at the search temperature, for the routing criterion."""
    pools = {}
    for path in sorted((POOL_DIR / f"T{temperature:g}" / split).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "greedy" in record:
                    pools[(record["dataset"], record["index"])] = record
    return pools


def load_split(split: str, rates: dict, size: str = "2b") -> dict:
    """Question embeddings, greedy labels and pass rates of every dataset, keeping questions that have a pass rate."""
    features, greedy, pass_rate, keys = [], [], [], []
    for name in DATASET_NAMES:
        cached = torch.load(FEATURE_DIR / f"{name}_{split}_{size}.pt", map_location="cpu", weights_only=True)
        for row, index in enumerate(cached["indices"].tolist()):
            if (name, index) in rates:
                features.append(cached["prompt_features"][row])
                greedy.append(float(cached["labels"][row]))
                pass_rate.append(rates[(name, index)])
                keys.append((name, index))
    greedy_t, rate_t = torch.tensor(greedy), torch.tensor(pass_rate)
    targets = {"greedy": greedy_t, "pass_rate": rate_t, "blend": 0.5 * (greedy_t + rate_t), "any_correct": (rate_t > 0).float()}
    return {"features": torch.stack(features).float(), "greedy": greedy_t, "pass_rate": rate_t, "targets": targets, "keys": keys}


def build(architecture: str, size: int, dropout: float) -> nn.Module:
    """A normalized linear or one-hidden-layer head over the question embedding."""
    if architecture == "linear":
        return nn.Sequential(nn.LayerNorm(size), nn.Dropout(dropout), nn.Linear(size, 1))
    hidden = int(architecture.removeprefix("mlp_"))
    return nn.Sequential(nn.LayerNorm(size), nn.Linear(size, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


def pass_rate_ranking(logits: torch.Tensor, rates: torch.Tensor, gap: float = 0.1) -> torch.Tensor:
    """Pairwise loss: a question with a clearly higher pass rate should score higher."""
    higher = (rates[:, None] - rates[None, :]) > gap
    if not higher.any():
        return logits.new_zeros(())
    return F.softplus(-(logits[:, None] - logits[None, :]))[higher].mean()


def gate_value(greedy: np.ndarray, scores: np.ndarray) -> float:
    """Early-stopping criterion: AURC of the greedy answers (lower is better)."""
    return aurc(greedy.astype(bool).tolist(), scores.tolist())["aurc"]


def train(config: dict, seed: int, train_data: dict, val_data: dict, device: torch.device,
          max_epochs: int, patience: int, batch_size: int) -> tuple[nn.Module, float, int]:
    """Train one configuration on one seed, keeping the epoch with the best validation AURC."""
    torch.manual_seed(seed)
    head = build(config["architecture"], train_data["features"].shape[1], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    features, target = train_data["features"].to(device), train_data["targets"][config["target"]].to(device)
    rates = train_data["pass_rate"].to(device)
    val_features, val_greedy = val_data["features"].to(device), val_data["greedy"].numpy()
    generator = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        for batch in torch.randperm(len(target), generator=generator).split(batch_size):
            batch = batch.to(device)
            logits = head(features[batch]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target[batch])
            if config["rank_weight"]:
                loss = loss + config["rank_weight"] * pass_rate_ranking(logits, rates[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            value = gate_value(val_greedy, head(val_features).squeeze(-1).cpu().numpy())
        if value < best:
            best, best_epoch, waited = value, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, best, best_epoch


def logits_of(head: nn.Module, data: dict, device: torch.device) -> np.ndarray:
    """Head logits for every question of a split."""
    with torch.no_grad():
        return head(data["features"].to(device)).squeeze(-1).cpu().numpy()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with average ranks for ties."""
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values))
        sorted_values = values[order]
        start = 0
        while start < len(values):
            stop = start
            while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
                stop += 1
            result[order[start:stop + 1]] = (start + stop) / 2
            start = stop + 1
        return result
    return float(np.corrcoef(ranks(np.asarray(a, float)), ranks(np.asarray(b, float)))[0, 1])


def evaluate(scores: np.ndarray, data: dict, pools: dict, search: dict | None) -> dict:
    """Test or validation metrics of one score, pooled and per dataset."""
    result = {}
    names = ["pooled", *DATASET_NAMES]
    for name in names:
        mask = np.array([name == "pooled" or key[0] == name for key in data["keys"]])
        if not mask.any():
            continue
        greedy, rates, s = data["greedy"].numpy()[mask], data["pass_rate"].numpy()[mask], scores[mask]
        row = {"questions": int(mask.sum()), "auroc_greedy": roc_auc(greedy, s), **aurc(greedy.astype(bool).tolist(), s.tolist()),
               "spearman_pass_rate": spearman(s, rates), "brier_pass_rate": float(np.mean((s - rates) ** 2))}
        if search is not None:
            keys = [key for key, keep in zip(data["keys"], mask) if keep]
            present = [i for i, key in enumerate(keys) if key in pools]
            if present:
                records = [pools[keys[i]] for i in present]
                row["routing"] = compute_curve(records, s[present].tolist(), search["k"], search["selector"])
        result[name] = row
    return result


def aggregate(rows: list[dict]):
    """Mean over seeds of nested metric dicts."""
    first = rows[0]
    if isinstance(first, dict):
        return {key: aggregate([row[key] for row in rows]) for key in first if all(key in row for row in rows)}
    return float(np.mean(rows))


def grid(quick: bool) -> list[dict]:
    """Every training configuration."""
    axes = {
        "target": TARGETS,
        "architecture": ("linear", "mlp_256") if quick else ("linear", "mlp_256", "mlp_512"),
        "learning_rate": (1e-3,) if quick else (3e-4, 1e-3),
        "weight_decay": (1e-4,) if quick else (1e-4, 1e-2),
        "dropout": (0.1,) if quick else (0.0, 0.3),
        "rank_weight": (0.0, 0.5),
    }
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def main() -> None:
    """Run the sweep and write results/pre_head_sweep.json and .md."""
    parser = argparse.ArgumentParser(description="Sweep the training of the pre-generation head.")
    parser.add_argument("--label-temperature", type=float, default=1.0)
    parser.add_argument("--label-samples", type=int, default=8)
    parser.add_argument("--search", type=Path, default=RESULTS_DIR / "candidate_search.json",
                        help="Chosen candidate search, for the routing criterion.")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "pre_head_sweep.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = {split: load_split(split, load_pass_rates(args.label_temperature, split, args.label_samples)) for split in ("train", "val", "test")}
    search = json.loads(args.search.read_text())["report"]["choice"] if args.search.exists() else None
    pools = {split: load_pools(split, float(search["temperature"])) if search else {} for split in ("val", "test")}
    print({split: len(d["keys"]) for split, d in data.items()}, "search:", search, flush=True)

    configs = []
    for number, config in enumerate(grid(args.quick), start=1):
        runs = []
        for seed in SEEDS:
            head, best, epoch = train(config, seed, data["train"], data["val"], device, args.max_epochs, args.patience, args.batch_size)
            val_logits, test_logits = logits_of(head, data["val"], device), logits_of(head, data["test"], device)
            temperature = fit_temperature(val_logits, data["val"]["greedy"].numpy())
            runs.append({"seed": seed, "epoch": epoch,
                         "val": evaluate(to_probabilities(val_logits, temperature), data["val"], pools["val"], search),
                         "test": evaluate(to_probabilities(test_logits, temperature), data["test"], pools["test"], search)})
        entry = {"config": config, "epochs": [r["epoch"] for r in runs],
                 "val": aggregate([r["val"] for r in runs]), "test": aggregate([r["test"] for r in runs])}
        configs.append(entry)
        print(f"[{number}] {config} val AURC {entry['val']['pooled']['aurc']:.4f} test AUROC {entry['test']['pooled']['auroc_greedy']:.4f}", flush=True)

    references = {}
    for label, file in REFERENCE_HEADS.items():
        head, spec, temperatures = load_head(HEADS / file)
        references[label] = {split: evaluate(score(head, spec, temperatures, data[split]["features"])["pre_generation" if spec.kind == "unified" else "main"],
                                              data[split], pools[split], search) for split in ("val", "test")}

    winners = {"gate": min(configs, key=lambda c: c["val"]["pooled"]["aurc"])}
    if search is not None:
        winners["routing"] = max(configs, key=lambda c: c["val"]["pooled"]["routing"]["mean_gain"])
    for target in TARGETS:
        winners[f"gate_{target}"] = min((c for c in configs if c["config"]["target"] == target), key=lambda c: c["val"]["pooled"]["aurc"])
    result = {"label_temperature": args.label_temperature, "label_samples": args.label_samples, "search": search,
              "questions": {split: len(d["keys"]) for split, d in data.items()},
              "winners": winners, "references": references, "configs": configs}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"[done] wrote {args.output} and {args.output.with_suffix('.md')}")


def markdown(result: dict) -> str:
    """Winners and references on test, pooled and per dataset."""
    lines = ["# Pre-generation head: training sweep", "",
             f"Input: the cached question embedding (constant across every row). Pass rate: first {result['label_samples']} samples "
             f"at T = {result['label_temperature']:g}. Winners chosen on validation; numbers are test means over three seeds.", "",
             "| Row | Config | AUROC (greedy) | AURC ↓ | E-AURC ↓ | Spearman with pass rate | Routing gain at 20% | Mean routing gain |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]

    def row(label, config, test):
        p = test["pooled"]
        routing = p.get("routing")
        cells = f"{100 * routing['0.2']['gain']:+.2f} | {100 * routing['mean_gain']:+.2f}" if routing else "– | –"
        lines.append(f"| {label} | {config} | {p['auroc_greedy']:.3f} | {p['aurc']:.4f} | {p['e_aurc']:.4f} | {p['spearman_pass_rate']:.3f} | {cells} |")

    for label, entry in result["references"].items():
        row(label, "reference", entry["test"])
    for name, winner in result["winners"].items():
        c = winner["config"]
        row(f"winner: {name}", f"{c['target']}, {c['architecture']}, lr {c['learning_rate']:g}, wd {c['weight_decay']:g}, "
                               f"dropout {c['dropout']:g}, rank {c['rank_weight']:g}", winner["test"])
    lines += ["", "## Per dataset (test AUROC against greedy correctness / AURC)", "",
              "| Row | " + " | ".join(DATASET_NAMES) + " |", "|---|" + "---:|" * len(DATASET_NAMES)]
    entries = [(label, entry["test"]) for label, entry in result["references"].items()]
    entries += [(f"winner: {name}", winner["test"]) for name, winner in result["winners"].items()]
    for label, test in entries:
        lines.append(f"| {label} | " + " | ".join(f"{test[n]['auroc_greedy']:.3f} / {test[n]['aurc']:.4f}" for n in DATASET_NAMES if n in test) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
