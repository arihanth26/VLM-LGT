# train_candidate_ranker.py
#
# The learned correction step: for each question, choose one answer from its candidate set (the
# greedy answer, sampled answers, and the answers of the correction strategies in
# correction_candidates.py). The correctness heads were trained to rank answers across questions;
# within one question they rank right above wrong only about two times in three, which is why
# best-of-k selection with them gained little. This ranker is trained for the within-question
# decision directly.
#
# Each distinct candidate is described by the constant cached question embedding, its own answer
# state (full image, original prompt, the answer), and scalar features: share of sampled answers
# that agree with it, whether it is the greedy answer, how many correction strategies proposed it,
# its teacher-forced answer probability, its length, and the head and probe scores. The loss is a
# softmax over the candidates of a question with every correct candidate as a target (only
# questions that have both a right and a wrong candidate carry signal), plus an optional BCE term.
# Training sets are resampled each epoch (pool size, with or without the correction candidates) so
# the vote features cover the sizes used at test time.
#
# Variants: full (states + scalars), scalars (no hidden states), states (no scalars). Each is swept
# over width, learning rate and the BCE weight on three seeds, the configuration is chosen on
# validation, and test is read once per chosen configuration. Reported for every candidate-set
# configuration (temperature, k, which corrections): accuracy, paired difference from greedy with
# a bootstrap interval and McNemar p, and the same for the untrained selectors on the same sets.
# Writes results/candidate_ranker.json and .md.

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.correction_stats import paired_difference
from experiments.approach1_latent_self_correction.scoring import normalize_vqa_answer

HERE = Path(__file__).parent
CORRECTION_DIR = HERE / "results" / "correction"
POOL_DIR = HERE / "results" / "pools"
FEATURE_DIR = HERE / "q_features"
RESULTS_DIR = HERE / "results"
SOURCES = ("vcd_0.5", "vcd_1", "zoom_0", "zoom_1", "zoom_2", "zoom_3")
CORRECTION_SETS = {"none": (), "vcd": ("vcd_0.5", "vcd_1"), "zoom": ("zoom_0", "zoom_1", "zoom_2", "zoom_3"), "all": SOURCES}
SCALARS = ("vote_share", "is_greedy", "correction_share", "tf_confidence", "log_tf_confidence", "tokens", "head", "probe", "head_logit")
SEEDS = (1, 2, 3)


def read_jsonl(path: Path) -> list[dict]:
    """Records of one JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_questions(split: str, include_extra: bool = False) -> list[dict]:
    """Every question of a split with its distinct candidates, their states and features, and its pools."""
    pools = {}
    temperatures = ("1",) if split == "train" else ("1", "1.3")
    for temperature in temperatures:
        for path in sorted((POOL_DIR / f"T{temperature}" / split).glob("*.jsonl")):
            for record in read_jsonl(path):
                if "samples" in record:
                    pools.setdefault((record["dataset"], record["index"]), {})[temperature] = record
    greedy_train = {}
    if split == "train":
        for name in DATASET_NAMES:
            for record in read_jsonl(RESULTS_DIR / f"{name}_train_2b_baseline.jsonl"):
                if "prediction" in record:
                    greedy_train[(name, record["index"])] = record["prediction"]["answer"]
    embeddings = {}
    for name in DATASET_NAMES:
        cached = torch.load(FEATURE_DIR / f"{name}_{split}_2b.pt", map_location="cpu", weights_only=True)
        for row, index in enumerate(cached["indices"].tolist()):
            embeddings[(name, index)] = cached["prompt_features"][row]

    questions: dict[tuple[str, int], dict] = {}
    # "full" shards (with corrections) are read last so they replace a states-only entry.
    for path in sorted((CORRECTION_DIR / split).glob("*.jsonl"), key=lambda p: ("_full_" in p.name, p.name)):
        saved = torch.load(path.with_suffix(".pt"), weights_only=False)["questions"]
        for record in read_jsonl(path):
            key = (record["dataset"], record["index"])
            if "encoded" not in record or record["index"] not in saved or key not in pools:
                continue
            stored = saved[record["index"]]
            greedy = greedy_train[key] if split == "train" else pools[key]["1"]["greedy"]["answer"]
            questions[key] = {
                "key": key, "answers": stored["answers"], "states": stored["states"], "encoded": record["encoded"],
                "question_state": embeddings[key], "greedy": greedy,
                "samples": {t: [c["answer"] for c in pools[key][t]["samples"]] for t in pools[key]},
                "corrections": {c["source"]: c["answer"] for c in record.get("corrections", [])},
            }
    if include_extra:
        add_extra_samples(questions, split)
    order = {name: i for i, name in enumerate(DATASET_NAMES)}
    return sorted(questions.values(), key=lambda q: (order[q["key"][0]], q["key"][1]))


def add_extra_samples(questions: dict, split: str) -> None:
    """Append the samples of extend_pools.py (pools past ten) and the states of their new answers."""
    for path in sorted((HERE / "results" / "correction_extra" / split).glob("*.jsonl")):
        saved = torch.load(path.with_suffix(".pt"), weights_only=False)["questions"]
        for record in read_jsonl(path):
            key = (record["dataset"], record["index"])
            if "samples" not in record or key not in questions:
                continue
            question = questions[key]
            new = saved.get(record["index"])
            if new is not None:
                missing = [i for i, answer in enumerate(new["answers"]) if answer not in question["encoded"]]
                question["answers"] = question["answers"] + [new["answers"][i] for i in missing]
                question["states"] = torch.cat([question["states"], new["states"][missing]]) if missing else question["states"]
                question["encoded"] = {**record["encoded"], **question["encoded"]}
            if all(c["answer"] in question["encoded"] for c in record["samples"]):
                question["samples"][record["temperature"]] = question["samples"][record["temperature"]][:10] + [c["answer"] for c in record["samples"]]


def candidate_set(question: dict, temperature: str, k: int, sources: tuple[str, ...]) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Distinct candidates of one configuration: row ids into the question's states, scalar features, and labels."""
    samples = question["samples"][temperature][:k]
    proposals = [question["corrections"][s] for s in sources if s in question["corrections"]]
    pool = list(dict.fromkeys([question["greedy"], *samples, *proposals]))
    position = {answer: i for i, answer in enumerate(question["answers"])}
    sample_votes = [normalize_vqa_answer(a) for a in samples]
    proposal_votes = [normalize_vqa_answer(a) for a in proposals]
    rows, features, labels = [], [], []
    for answer in pool:
        info = question["encoded"][answer]
        norm = normalize_vqa_answer(answer)
        head = min(max(info["head_confidence"], 1e-6), 1 - 1e-6)
        rows.append(position[answer])
        features.append([
            sample_votes.count(norm) / max(1, len(samples)), float(answer == question["greedy"]),
            proposal_votes.count(norm) / max(1, len(proposals)) if proposals else 0.0,
            info["tf_confidence"], float(np.log(max(info["tf_confidence"], 1e-6))), info["tokens"] / 10.0,
            info["head_confidence"], info["probe_confidence"], float(np.log(head / (1 - head))),
        ])
        labels.append(float(info["correct"]))
    return rows, np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


class Ranker(nn.Module):
    """Scores one candidate from the question embedding, its answer state, and scalar features."""

    def __init__(self, variant: str, hidden: int, state_size: int = 2048):
        super().__init__()
        self.variant = variant
        self.question_norm, self.answer_norm = nn.LayerNorm(state_size), nn.LayerNorm(state_size)
        inputs = (0 if variant == "scalars" else 2 * state_size) + (0 if variant == "states" else len(SCALARS))
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, question: torch.Tensor, answer: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.variant != "scalars":
            parts += [self.question_norm(question.float()), self.answer_norm(answer.float())]
        if self.variant != "states":
            parts.append(scalars)
        return self.network(torch.cat(parts, dim=-1)).squeeze(-1)


def batch_tensors(questions: list[dict], sets: list[tuple], device) -> dict:
    """Pad a batch of candidate sets to the largest one."""
    width = max(len(rows) for rows, _, _ in sets)
    count = len(sets)
    answer = torch.zeros(count, width, 2048, dtype=torch.float16)
    scalars = torch.zeros(count, width, len(SCALARS))
    labels = torch.zeros(count, width)
    mask = torch.zeros(count, width, dtype=torch.bool)
    for i, (question, (rows, features, label)) in enumerate(zip(questions, sets)):
        answer[i, :len(rows)] = question["states"][rows]
        scalars[i, :len(rows)] = torch.from_numpy(features)
        labels[i, :len(rows)] = torch.from_numpy(label)
        mask[i, :len(rows)] = True
    question_state = torch.stack([q["question_state"] for q in questions]).unsqueeze(1).expand(-1, width, -1)
    return {"question": question_state.to(device), "answer": answer.to(device), "scalars": scalars.to(device),
            "labels": labels.to(device), "mask": mask.to(device)}


def listwise_loss(scores: torch.Tensor, batch: dict, bce_weight: float) -> torch.Tensor:
    """Multi-positive softmax over each question's candidates, on questions with a right and a wrong candidate."""
    masked = scores.masked_fill(~batch["mask"], float("-inf"))
    positives = batch["labels"].bool() & batch["mask"]
    mixed = positives.any(1) & (~positives & batch["mask"]).any(1)
    loss = scores.new_zeros(())
    if mixed.any():
        all_terms = torch.logsumexp(masked[mixed], dim=1)
        right_terms = torch.logsumexp(masked[mixed].masked_fill(~positives[mixed], float("-inf")), dim=1)
        loss = (all_terms - right_terms).mean()
    if bce_weight:
        bce = nn.functional.binary_cross_entropy_with_logits(scores[batch["mask"]], batch["labels"][batch["mask"]])
        loss = loss + bce_weight * bce
    return loss


def predict(models: list[Ranker], questions: list[dict], configuration: tuple, device, batch_size: int = 256) -> list[bool]:
    """Whether the top candidate is correct, per question, scoring with the mean of the given rankers."""
    temperature, k, sources = configuration
    chosen = []
    for model in models:
        model.eval()
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            part = questions[start:start + batch_size]
            sets = [candidate_set(q, temperature, k, sources) for q in part]
            batch = batch_tensors(part, sets, device)
            scores = torch.stack([m(batch["question"], batch["answer"], batch["scalars"]) for m in models]).mean(0)
            top = scores.masked_fill(~batch["mask"], float("-inf")).argmax(1).cpu().numpy()
            chosen += [bool(labels[i]) for (_, _, labels), i in zip(sets, top)]
    return chosen


def untrained_selectors(questions: list[dict], configuration: tuple) -> dict[str, list[bool]]:
    """Greedy, head best, vote with head tie-break, and oracle on the same candidate sets."""
    temperature, k, sources = configuration
    result = {"greedy": [], "head_best": [], "vote_then_head": [], "oracle": []}
    for q in questions:
        _, features, labels = candidate_set(q, temperature, k, sources)
        result["greedy"].append(bool(labels[int(np.argmax(features[:, 1]))]))
        result["head_best"].append(bool(labels[int(np.argmax(features[:, 6]))]))
        result["vote_then_head"].append(bool(labels[int(np.lexsort((features[:, 6], features[:, 0]))[-1])]))
        result["oracle"].append(bool(labels.max() > 0))
    return result


def train(variant: str, hidden: int, learning_rate: float, bce_weight: float, seed: int, train_q: list[dict],
          val_q: list[dict], device, epochs: int, batch_size: int, select_config: tuple) -> tuple[Ranker, float]:
    """Train one ranker, keeping the epoch with the best validation accuracy on the selection configuration."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Ranker(variant, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    has_corrections = [bool(q["corrections"]) for q in train_q]
    cache: dict = {}
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_q))
        for start in range(0, len(order), batch_size):
            part = [train_q[i] for i in order[start:start + batch_size]]
            sets = []
            for i, q in zip(order[start:start + batch_size], part):
                k = int(rng.choice((3, 5, 8)))
                sources = SOURCES if has_corrections[i] and rng.random() < 0.7 else ()
                if (i, k, bool(sources)) not in cache:
                    cache[(i, k, bool(sources))] = candidate_set(q, "1", k, sources)
                sets.append(cache[(i, k, bool(sources))])
            batch = batch_tensors(part, sets, device)
            loss = listwise_loss(model(batch["question"], batch["answer"], batch["scalars"]), batch, bce_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = float(np.mean(predict([model], val_q, select_config, device)))
        if accuracy > best:
            best, best_state = accuracy, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def configurations(test_like: list[dict]) -> list[tuple]:
    """Every (temperature, k, corrections) set evaluated."""
    has_corrections = any(q["corrections"] for q in test_like)
    names = CORRECTION_SETS if has_corrections else {"none": ()}
    return [(t, k, (name, sources)) for t in ("1", "1.3") for k in (0, 3, 5, 7, 10) for name, sources in names.items()
            if not (k == 0 and name == "none")]


def report_rows(outcomes: dict[str, list[bool]], greedy: list[bool], questions: list[dict]) -> dict:
    """Pooled and per-dataset accuracy and paired difference from greedy of every selector."""
    rows = {}
    for name, chosen in outcomes.items():
        entry = {"pooled": {**paired_difference(greedy, chosen), "accuracy": float(np.mean(chosen))}}
        for dataset in DATASET_NAMES:
            mask = [q["key"][0] == dataset for q in questions]
            g = [x for x, m in zip(greedy, mask) if m]
            c = [x for x, m in zip(chosen, mask) if m]
            entry[dataset] = {**paired_difference(g, c), "accuracy": float(np.mean(c)), "greedy": float(np.mean(g))}
        rows[name] = entry
    return rows


def main() -> None:
    """Sweep the ranker variants, choose on validation, and report every configuration on test."""
    parser = argparse.ArgumentParser(description="Train the within-question candidate ranker.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "candidate_ranker.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {split: load_questions(split) for split in ("train", "val", "test")}
    print({split: len(rows) for split, rows in data.items()}, flush=True)

    select_config = ("1", 10, CORRECTION_SETS["all"] if any(q["corrections"] for q in data["val"]) else ())
    grid = list(itertools.product(("full", "scalars", "states"), (256,) if args.quick else (512,),
                                  (1e-3,) if args.quick else (3e-4, 1e-3), (0.0, 0.3)))
    sweep, models = [], {}
    for variant, hidden, rate, bce in grid:
        runs = [train(variant, hidden, rate, bce, seed, data["train"], data["val"], device, args.epochs, args.batch_size, select_config)
                for seed in SEEDS]
        entry = {"variant": variant, "hidden": hidden, "learning_rate": rate, "bce_weight": bce,
                 "val_accuracy": float(np.mean([accuracy for _, accuracy in runs]))}
        sweep.append(entry)
        models[(variant, hidden, rate, bce)] = [model for model, _ in runs]
        print(entry, flush=True)

    winners = {}
    for variant in ("full", "scalars", "states"):
        best = max((e for e in sweep if e["variant"] == variant), key=lambda e: e["val_accuracy"])
        winners[variant] = {"config": best, "models": models[(variant, best["hidden"], best["learning_rate"], best["bce_weight"])]}

    results = {"questions": {s: len(r) for s, r in data.items()}, "sweep": sweep, "select_config": list(select_config[:2]),
               "val": {}, "test": {}}
    for split in ("val", "test"):
        for temperature, k, (name, sources) in configurations(data[split]):
            config = (temperature, k, sources)
            outcomes = untrained_selectors(data[split], config)
            for variant, winner in winners.items():
                # The reported ranker averages the scores of its three seeds; single-seed accuracies are kept for the spread.
                outcomes[f"ranker_{variant}"] = predict(winner["models"], data[split], config, device)
                results.setdefault("seed_accuracy", {}).setdefault(split, {})[f"T{temperature}_k{k}_{name}_{variant}"] = [
                    float(np.mean(predict([model], data[split], config, device))) for model in winner["models"]]
            results[split][f"T{temperature}_k{k}_{name}"] = report_rows(outcomes, outcomes["greedy"], data[split])
        print(f"[evaluated] {split}", flush=True)

    # Final choice on validation: the configuration and variant with the best validation accuracy.
    best_key, best_variant = max(((key, v) for key in results["val"] for v in ("full", "scalars", "states")),
                                 key=lambda item: results["val"][item[0]][f"ranker_{item[1]}"]["pooled"]["accuracy"])
    results["choice"] = {"configuration": best_key, "variant": best_variant,
                         "val": results["val"][best_key][f"ranker_{best_variant}"]["pooled"],
                         "test": results["test"][best_key]}
    results["winners"] = {variant: winner["config"] for variant, winner in winners.items()}
    args.output.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(results), encoding="utf-8")
    torch.save({variant: [m.state_dict() for m in winner["models"]] for variant, winner in winners.items()},
               RESULTS_DIR / "candidate_ranker_models.pt")
    print(f"[done] wrote {args.output} and {args.output.with_suffix('.md')}")


def markdown(results: dict) -> str:
    """Chosen configuration on test, per dataset, and the test table for every configuration."""
    choice = results["choice"]
    test = choice["test"]
    variant = choice["variant"]
    lines = ["# Learned candidate ranker (correction by selection)", "",
             f"Chosen on validation: `{choice['configuration']}`, ranker variant `{variant}` "
             f"(validation gain over greedy {100 * choice['val']['difference']:+.2f} points). Test is read once for it.", "",
             "| Dataset | Greedy | Ranker | Difference (95% interval) | Fixed / broken | p | Head best | Vote then head | Oracle |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for dataset in ("pooled", *DATASET_NAMES):
        r = test[f"ranker_{variant}"][dataset]
        greedy = test["greedy"][dataset]["accuracy"]
        lines.append(f"| {dataset} | {greedy:.2%} | {r['accuracy']:.2%} | {100 * r['difference']:+.2f} "
                     f"({100 * r['ci_low']:+.2f} to {100 * r['ci_high']:+.2f}) | +{r['fixed']} / -{r['broken']} | {r['p_value']:.4f} | "
                     f"{test['head_best'][dataset]['accuracy']:.2%} | {test['vote_then_head'][dataset]['accuracy']:.2%} | {test['oracle'][dataset]['accuracy']:.2%} |")
    lines += ["", "## Every configuration on test (pooled accuracy; greedy is the same answer in every row)", "",
              "| Configuration | Greedy | Head best | Vote then head | Ranker full | Ranker scalars | Ranker states | Oracle |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, rows in results["test"].items():
        lines.append(f"| {key} | " + " | ".join(f"{rows[n]['pooled']['accuracy']:.2%}" for n in
                     ("greedy", "head_best", "vote_then_head", "ranker_full", "ranker_scalars", "ranker_states", "oracle")) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
