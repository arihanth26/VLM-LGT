# train_ranker_v2.py
#
# Second version of the within-question candidate ranker (train_candidate_ranker.py is the first), with
# the outputs of correction_v2.py:
#   new candidates   grounded crop, grounded crop with the full image as context, step-by-step reasoning
#   new feature      the model's own judgement of each candidate, P(Yes) to "is this answer correct?"
#   more training    every correction source on 12,000 training questions (the first ranker saw
#                    contrastive and crop candidates on 3,200)
# It is trained only on training questions that have the second-round outputs, so every training set
# can include every source. Variants: full (states and the first ranker's scalars), full_judge (plus the
# judge), scalars_judge (scalars and judge, no states), judge_only (the judge score alone, untrained).
# The first ranker is scored on the same candidate sets for comparison. Configurations (k, which
# corrections) and the variant are chosen on validation; test is read once for the choice.
# Writes results/candidate_ranker_v2.json, .md, and candidate_ranker_v2_models.pt.

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
from experiments.approach1_latent_self_correction.train_candidate_ranker import (
    SCALARS,
    Ranker,
    listwise_loss,
    load_questions,
    read_jsonl,
)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
V2_DIR = RESULTS / "correction_v2"
FIRST = ("vcd_0.5", "vcd_1", "zoom_0", "zoom_1", "zoom_2", "zoom_3")
SETS = {
    "none": (), "zoom": FIRST[2:], "vcd": FIRST[:2], "first_round": FIRST, "ground": ("ground", "ground_ctx"), "cot": ("cot",),
    "ground_cot": ("ground", "ground_ctx", "cot"), "zoom_ground": FIRST[2:] + ("ground", "ground_ctx"),
    "everything": FIRST + ("ground", "ground_ctx", "cot"),
}
SCALARS_V2 = SCALARS + ("judge", "judge_logit")
VARIANTS = ("full", "full_judge", "scalars_judge")
SEEDS = (1, 2, 3)


def merge_v2(questions: list[dict], split: str) -> list[dict]:
    """Add second-round corrections, encodings, states and judge scores; keep only questions that have them."""
    by_key = {q["key"]: q for q in questions}
    kept = {}
    for path in sorted((V2_DIR / split).glob("*.jsonl")):
        saved = torch.load(path.with_suffix(".pt"), weights_only=False)["questions"]
        for record in read_jsonl(path):
            key = (record["dataset"], record["index"])
            if "judge" not in record or key not in by_key:
                continue
            q = by_key[key]
            new = saved.get(record["index"])
            if new is not None:
                missing = [i for i, a in enumerate(new["answers"]) if a not in q["encoded"]]
                q["answers"] = q["answers"] + [new["answers"][i] for i in missing]
                if missing:
                    q["states"] = torch.cat([q["states"], new["states"][missing]])
            q["encoded"] = {**record["encoded"], **q["encoded"]}
            q["corrections"] = {**q["corrections"], **{c["source"]: c["answer"] for c in record["corrections"]}}
            q["judge"] = record["judge"]
            if all(a in q["encoded"] for a in q["corrections"].values()):
                kept[key] = q
    order = {name: i for i, name in enumerate(DATASET_NAMES)}
    return sorted(kept.values(), key=lambda q: (order[q["key"][0]], q["key"][1]))


def candidate_set(q: dict, temperature: str, k: int, sources: tuple) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Distinct candidates with the first ranker's nine scalars followed by the judge and its logit."""
    samples = q["samples"][temperature][:k]
    proposals = [q["corrections"][s] for s in sources if s in q["corrections"]]
    pool = list(dict.fromkeys([q["greedy"], *samples, *proposals]))
    position = {a: i for i, a in enumerate(q["answers"])}
    sample_votes = [normalize_vqa_answer(a) for a in samples]
    proposal_votes = [normalize_vqa_answer(a) for a in proposals]
    rows, features, labels = [], [], []
    for answer in pool:
        info = q["encoded"][answer]
        norm = normalize_vqa_answer(answer)
        head = min(max(info["head_confidence"], 1e-6), 1 - 1e-6)
        judge = min(max(q["judge"].get(answer, 0.5), 1e-6), 1 - 1e-6)
        rows.append(position[answer])
        features.append([
            sample_votes.count(norm) / max(1, len(samples)), float(answer == q["greedy"]),
            proposal_votes.count(norm) / max(1, len(proposals)) if proposals else 0.0,
            info["tf_confidence"], float(np.log(max(info["tf_confidence"], 1e-6))), info["tokens"] / 10.0,
            info["head_confidence"], info["probe_confidence"], float(np.log(head / (1 - head))),
            judge, float(np.log(judge / (1 - judge))),
        ])
        labels.append(float(info["correct"]))
    return rows, np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


class RankerV2(nn.Module):
    """The first ranker's network with the judge features optional."""

    def __init__(self, variant: str, hidden: int, state_size: int = 2048):
        super().__init__()
        self.variant = variant
        self.use_states = variant != "scalars_judge"
        self.scalars = len(SCALARS_V2) if variant.endswith("judge") else len(SCALARS)
        self.question_norm, self.answer_norm = nn.LayerNorm(state_size), nn.LayerNorm(state_size)
        inputs = (2 * state_size if self.use_states else 0) + self.scalars
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, question, answer, scalars):
        parts = [self.question_norm(question.float()), self.answer_norm(answer.float())] if self.use_states else []
        parts.append(scalars[..., : self.scalars])
        return self.network(torch.cat(parts, dim=-1)).squeeze(-1)


def batch_tensors(questions: list[dict], sets: list[tuple], device) -> dict:
    """Pad a batch of candidate sets."""
    width = max(len(r) for r, _, _ in sets)
    n = len(sets)
    answer = torch.zeros(n, width, 2048, dtype=torch.float16)
    scalars = torch.zeros(n, width, len(SCALARS_V2))
    labels = torch.zeros(n, width)
    mask = torch.zeros(n, width, dtype=torch.bool)
    for i, (q, (rows, features, label)) in enumerate(zip(questions, sets)):
        answer[i, :len(rows)] = q["states"][rows]
        scalars[i, :len(rows)] = torch.from_numpy(features)
        labels[i, :len(rows)] = torch.from_numpy(label)
        mask[i, :len(rows)] = True
    question = torch.stack([q["question_state"] for q in questions]).unsqueeze(1).expand(-1, width, -1)
    return {"question": question.to(device), "answer": answer.to(device), "scalars": scalars.to(device),
            "labels": labels.to(device), "mask": mask.to(device)}


def choose(models: list, questions: list[dict], config: tuple, device, first_ranker: bool = False, batch_size: int = 256) -> list[bool]:
    """Whether the top candidate under the seed-averaged scores is correct, per question."""
    temperature, k, sources = config
    chosen = []
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            part = questions[start:start + batch_size]
            sets = [candidate_set(q, temperature, k, sources) for q in part]
            batch = batch_tensors(part, sets, device)
            scalars = batch["scalars"][..., : len(SCALARS)] if first_ranker else batch["scalars"]
            scores = torch.stack([m(batch["question"], batch["answer"], scalars) for m in models]).mean(0)
            top = scores.masked_fill(~batch["mask"], float("-inf")).argmax(1).cpu().numpy()
            chosen += [bool(labels[i]) for (_, _, labels), i in zip(sets, top)]
    return chosen


def untrained(questions: list[dict], config: tuple) -> dict[str, list[bool]]:
    """Greedy, head best, vote then head, the judge alone, and the oracle on the same sets."""
    temperature, k, sources = config
    out = {"greedy": [], "head_best": [], "vote_then_head": [], "judge_best": [], "oracle": []}
    for q in questions:
        _, f, labels = candidate_set(q, temperature, k, sources)
        out["greedy"].append(bool(labels[int(np.argmax(f[:, 1]))]))
        out["head_best"].append(bool(labels[int(np.argmax(f[:, 6]))]))
        out["vote_then_head"].append(bool(labels[int(np.lexsort((f[:, 6], f[:, 0]))[-1])]))
        out["judge_best"].append(bool(labels[int(np.argmax(f[:, 9] + 1e-3 * f[:, 1]))]))
        out["oracle"].append(bool(labels.max() > 0))
    return out


def train(variant: str, rate: float, seed: int, train_q: list[dict], val_q: list[dict], device, epochs: int,
          batch_size: int, select: tuple, hidden: int = 512) -> tuple[RankerV2, float]:
    """One ranker on one seed, keeping the epoch with the best validation accuracy on the selection configuration."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RankerV2(variant, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-2)
    cache, best, best_state = {}, -1.0, None
    names = list(SETS)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_q))
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            sets = []
            for i in ids:
                key = (int(i), int(rng.choice((0, 3, 5, 8))), names[int(rng.integers(len(names)))])
                if key not in cache:
                    cache[key] = candidate_set(train_q[i], "1", key[1], SETS[key[2]])
                sets.append(cache[key])
            batch = batch_tensors([train_q[i] for i in ids], sets, device)
            loss = listwise_loss(model(batch["question"], batch["answer"], batch["scalars"]), batch, 0.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        accuracy = float(np.mean(choose([model], val_q, select, device)))
        if accuracy > best:
            best, best_state = accuracy, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def first_rankers(device) -> list[Ranker]:
    """The three seeds of the first ranker's states-and-scalars variant."""
    winners = json.loads((RESULTS / "candidate_ranker.json").read_text())["winners"]
    states = torch.load(RESULTS / "candidate_ranker_models.pt", map_location=device, weights_only=False)["full"]
    models = []
    for state in states:
        model = Ranker("full", winners["full"]["hidden"]).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def rows_for(questions: list[dict], outcomes: dict[str, list[bool]]) -> dict:
    """Pooled and per-dataset accuracy and paired difference from greedy for every selector."""
    greedy = outcomes["greedy"]
    result = {}
    for name, chosen in outcomes.items():
        entry = {"pooled": {**paired_difference(greedy, chosen), "accuracy": float(np.mean(chosen)), "greedy": float(np.mean(greedy))}}
        for dataset in DATASET_NAMES:
            mask = [q["key"][0] == dataset for q in questions]
            g, c = [x for x, m in zip(greedy, mask) if m], [x for x, m in zip(chosen, mask) if m]
            if g:
                entry[dataset] = {**paired_difference(g, c), "accuracy": float(np.mean(c)), "greedy": float(np.mean(g))}
        result[name] = entry
    return result


def source_accuracy(questions: list[dict]) -> dict:
    """How often each correction source's answer is right on its own, and how often it differs from greedy."""
    table = {}
    for source in SETS["everything"]:
        right = differs = count = 0
        for q in questions:
            answer = q["corrections"].get(source)
            if answer is None:
                continue
            count += 1
            right += int(q["encoded"][answer]["correct"])
            differs += int(answer != q["greedy"])
        table[source] = {"questions": count, "accuracy": right / max(1, count), "differs_from_greedy": differs / max(1, count)}
    return table


def main() -> None:
    """Train the variants, compare with the first ranker, choose on validation, report on test."""
    parser = argparse.ArgumentParser(description="Train the second candidate ranker.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=RESULTS / "candidate_ranker_v2.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {split: merge_v2(load_questions(split), split) for split in ("train", "val", "test")}
    print({s: len(q) for s, q in data.items()}, flush=True)
    select = ("1", 5, SETS["everything"])

    sweep, models = [], {}
    for variant, rate in itertools.product(VARIANTS, (3e-4, 1e-3)):
        runs = [train(variant, rate, seed, data["train"], data["val"], device, args.epochs, args.batch_size, select) for seed in SEEDS]
        sweep.append({"variant": variant, "learning_rate": rate, "val_accuracy": float(np.mean([a for _, a in runs]))})
        models[(variant, rate)] = [m for m, _ in runs]
        print(sweep[-1], flush=True)
    winners = {v: max((e for e in sweep if e["variant"] == v), key=lambda e: e["val_accuracy"]) for v in VARIANTS}
    chosen_models = {v: models[(v, winners[v]["learning_rate"])] for v in VARIANTS}
    first = first_rankers(device)

    result = {"questions": {s: len(q) for s, q in data.items()}, "sweep": sweep, "winners": winners,
              "source_accuracy": {s: source_accuracy(data[s]) for s in ("val", "test")}, "val": {}, "test": {}}
    configs = [("1", k, name) for k in (0, 5) for name in SETS if not (k == 0 and name == "none")]
    for split in ("val", "test"):
        for temperature, k, name in configs:
            config = (temperature, k, SETS[name])
            outcomes = untrained(data[split], config)
            outcomes["ranker_v1"] = choose(first, data[split], config, device, first_ranker=True)
            for variant, ms in chosen_models.items():
                outcomes[f"ranker_v2_{variant}"] = choose(ms, data[split], config, device)
            result[split][f"T{temperature}_k{k}_{name}"] = rows_for(data[split], outcomes)
        print(f"[evaluated] {split}", flush=True)
    candidates = [(key, sel) for key in result["val"] for sel in result["val"][key] if sel.startswith("ranker")]
    best_key, best_sel = max(candidates, key=lambda c: result["val"][c[0]][c[1]]["pooled"]["accuracy"])
    result["choice"] = {"configuration": best_key, "selector": best_sel, "val": result["val"][best_key][best_sel]["pooled"],
                        "test": result["test"][best_key]}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    torch.save({v: [m.state_dict() for m in ms] for v, ms in chosen_models.items()}, RESULTS / "candidate_ranker_v2_models.pt")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"[done] wrote {args.output}")


def markdown(result: dict) -> str:
    """Chosen configuration per dataset, then every configuration on test."""
    c = result["choice"]
    test = c["test"]
    lines = ["# Candidate ranker v2: grounded crops, reasoning, self-judge, more training", "",
             f"Chosen on validation: `{c['configuration']}` with `{c['selector']}` (validation gain {100 * c['val']['difference']:+.2f} points). "
             f"Questions: {result['questions']}.", "",
             "| Dataset | Greedy | Chosen | Difference (95% interval) | Fixed / broken | p | Ranker v1, same set | Oracle |", "|---|---:|---:|---|---|---:|---:|---:|"]
    for d in ("pooled", *DATASET_NAMES):
        r = test[c["selector"]][d]
        lines.append(f"| {d} | {100 * r['greedy']:.2f}% | {100 * r['accuracy']:.2f}% | {100 * r['difference']:+.2f} ({100 * r['ci_low']:+.2f} to {100 * r['ci_high']:+.2f}) | "
                     f"+{r['fixed']} / -{r['broken']} | {r['p_value']:.4f} | {100 * test['ranker_v1'][d]['accuracy']:.2f}% | {100 * test['oracle'][d]['accuracy']:.2f}% |")
    names = ["greedy", "head_best", "vote_then_head", "judge_best", "ranker_v1"] + [f"ranker_v2_{v}" for v in VARIANTS] + ["oracle"]
    lines += ["", "## Every configuration on test (pooled accuracy)", "", "| Configuration | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key, rows in result["test"].items():
        lines.append(f"| {key} | " + " | ".join(f"{100 * rows[n]['pooled']['accuracy']:.2f}%" for n in names) + " |")
    lines += ["", "## Correction sources on their own (test)", "", "| Source | Questions | Accuracy | Differs from greedy |", "|---|---:|---:|---:|"]
    for source, row in result["source_accuracy"]["test"].items():
        lines.append(f"| {source} | {row['questions']:,} | {100 * row['accuracy']:.2f}% | {100 * row['differs_from_greedy']:.1f}% |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
