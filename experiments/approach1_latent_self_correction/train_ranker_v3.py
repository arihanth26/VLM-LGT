# train_ranker_v3.py
#
# Third version of the within-question candidate ranker: one ranker over greedy, the step-by-step
# reasoning answer and the zoomed crops (optionally the grounded crops), that learns which source to
# trust on which dataset.
#
# Why: the two correction rounds are good at different things. Reasoning (round 2) fixes ChartQA and
# ScienceQA; the corner crops (round 1) are better for TextVQA and DocVQA. The v2 ranker over every
# source ("everything") does not exploit this: it sees no feature saying where a candidate came from or
# which dataset the question is from.
#
# What is new over train_ranker_v2.py (which is reused unchanged for loading and the v2 features):
#   source features   for each correction group (zoom, vcd, ground, cot), the share of that group's
#                     offered proposals that agree with the candidate (0 when the group is not offered)
#   dataset feature   one-hot of the question's dataset
#   candidate sets    zoom_cot (crops + reasoning) and zoom_ground_cot
#   router baseline   per dataset, the (configuration, selector) with the best validation accuracy,
#                     applied to test. Untrained, chosen on validation only.
# No new generation: every candidate is already encoded in results/correction and results/correction_v2.
# Variants: full_ds (states + every scalar) and scalars_ds (no states). The v1 and v2 rankers are
# scored on the same sets for comparison. Configurations and selectors are chosen on validation; test is
# read once for the choice.
# Writes results/candidate_ranker_v3.json, .md and candidate_ranker_v3_models.pt.

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.scoring import normalize_vqa_answer
from experiments.approach1_latent_self_correction.train_candidate_ranker import listwise_loss, load_questions
from experiments.approach1_latent_self_correction.train_ranker_v2 import (
    FIRST,
    RESULTS,
    SCALARS_V2,
    SEEDS,
    RankerV2,
    candidate_set as candidate_set_v2,
    choose as choose_v2,
    first_rankers,
    merge_v2,
    rows_for,
    source_accuracy,
    untrained,
)
from experiments.approach1_latent_self_correction.train_ranker_v2 import SETS as SETS_V2
from experiments.approach1_latent_self_correction.train_ranker_v2 import VARIANTS as VARIANTS_V2

GROUPS = {"zoom": FIRST[2:], "vcd": FIRST[:2], "ground": ("ground", "ground_ctx"), "cot": ("cot",)}
SETS = {
    **SETS_V2,
    "zoom_cot": GROUPS["zoom"] + GROUPS["cot"],
    "zoom_ground_cot": GROUPS["zoom"] + GROUPS["ground"] + GROUPS["cot"],
}
SCALARS_V3 = SCALARS_V2 + tuple(f"from_{g}" for g in GROUPS) + tuple(f"dataset_{d}" for d in DATASET_NAMES)
VARIANTS = ("full_ds", "scalars_ds")
SELECT = ("1", 0, SETS["zoom_cot"])


def candidate_set(q: dict, temperature: str, k: int, sources: tuple) -> tuple[list[int], np.ndarray, np.ndarray]:
    """The v2 candidate set and features, followed by the source-agreement and dataset features."""
    rows, features, labels = candidate_set_v2(q, temperature, k, sources)
    offered = {g: [normalize_vqa_answer(q["corrections"][s]) for s in members if s in sources and s in q["corrections"]]
               for g, members in GROUPS.items()}
    dataset = [float(q["key"][0] == d) for d in DATASET_NAMES]
    extra = []
    for row in rows:
        norm = normalize_vqa_answer(q["answers"][row])
        extra.append([votes.count(norm) / len(votes) if votes else 0.0 for votes in offered.values()] + dataset)
    extra = np.asarray(extra, dtype=np.float32).reshape(len(rows), len(GROUPS) + len(DATASET_NAMES))
    return rows, np.concatenate([features, extra], axis=1), labels


class RankerV3(nn.Module):
    """The v2 network with the source and dataset features; scalars_ds drops the hidden states."""

    def __init__(self, variant: str, hidden: int, state_size: int = 2048):
        super().__init__()
        self.variant = variant
        self.use_states = variant == "full_ds"
        self.question_norm, self.answer_norm = nn.LayerNorm(state_size), nn.LayerNorm(state_size)
        inputs = (2 * state_size if self.use_states else 0) + len(SCALARS_V3)
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, question, answer, scalars):
        parts = [self.question_norm(question.float()), self.answer_norm(answer.float())] if self.use_states else []
        parts.append(scalars)
        return self.network(torch.cat(parts, dim=-1)).squeeze(-1)


def batch_tensors(questions: list[dict], sets: list[tuple], device) -> dict:
    """Pad a batch of candidate sets (v2's helper with the wider scalar block)."""
    width = max(len(r) for r, _, _ in sets)
    n = len(sets)
    answer = torch.zeros(n, width, 2048, dtype=torch.float16)
    scalars = torch.zeros(n, width, len(SCALARS_V3))
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


def choose(models: list, questions: list[dict], config: tuple, device, batch_size: int = 256) -> list[bool]:
    """Whether the top candidate under the seed-averaged v3 scores is correct, per question."""
    temperature, k, sources = config
    chosen = []
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            part = questions[start:start + batch_size]
            sets = [candidate_set(q, temperature, k, sources) for q in part]
            batch = batch_tensors(part, sets, device)
            scores = torch.stack([m(batch["question"], batch["answer"], batch["scalars"]) for m in models]).mean(0)
            top = scores.masked_fill(~batch["mask"], float("-inf")).argmax(1).cpu().numpy()
            chosen += [bool(labels[i]) for (_, _, labels), i in zip(sets, top)]
    return chosen


def train(variant: str, rate: float, seed: int, train_q: list[dict], val_q: list[dict], device, epochs: int,
          batch_size: int, hidden: int = 512) -> tuple[RankerV3, float]:
    """One ranker on one seed; random k and candidate set per question each step, best validation epoch kept."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RankerV3(variant, hidden).to(device)
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
        accuracy = float(np.mean(choose([model], val_q, SELECT, device)))
        if accuracy > best:
            best, best_state = accuracy, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def v2_rankers(device) -> dict[str, list[RankerV2]]:
    """The saved v2 rankers (three seeds per variant), or none if the file is missing."""
    path = RESULTS / "candidate_ranker_v2_models.pt"
    if not path.exists():
        print(f"[warn] {path} not found; skipping the v2 comparison", flush=True)
        return {}
    saved = torch.load(path, map_location=device, weights_only=False)
    out = {}
    for variant in VARIANTS_V2:
        models = []
        for state in saved.get(variant, []):
            model = RankerV2(variant, state["network.0.weight"].shape[0]).to(device)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        if models:
            out[variant] = models
    return out


def router(questions: dict[str, list[dict]], outcomes: dict[str, dict[tuple, list[bool]]]) -> tuple[dict, dict[str, list[bool]]]:
    """Per dataset, the (configuration, selector) best on validation; returns the choices and the routed outcomes."""
    choices, routed = {}, {}
    for split in ("val", "test"):
        routed[split] = [False] * len(questions[split])
    for dataset in DATASET_NAMES:
        val_ids = [i for i, q in enumerate(questions["val"]) if q["key"][0] == dataset]
        if not val_ids:
            continue
        best = max(outcomes["val"], key=lambda c: (np.mean([outcomes["val"][c][i] for i in val_ids]), c[1] != "greedy"))
        choices[dataset] = {"configuration": best[0], "selector": best[1],
                            "val_accuracy": float(np.mean([outcomes["val"][best][i] for i in val_ids]))}
        for split in ("val", "test"):
            for i, q in enumerate(questions[split]):
                if q["key"][0] == dataset:
                    routed[split][i] = outcomes[split][best][i]
    return choices, routed


def main() -> None:
    """Train the v3 variants, score v1/v2/v3 and the router on every configuration, choose on validation."""
    parser = argparse.ArgumentParser(description="Train the third candidate ranker (reasoning + crops, source and dataset features).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rates", type=float, nargs="+", default=[3e-4, 1e-3])
    parser.add_argument("--output", type=Path, default=RESULTS / "candidate_ranker_v3.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {split: merge_v2(load_questions(split), split) for split in ("train", "val", "test")}
    print({s: len(q) for s, q in data.items()}, flush=True)

    sweep, models = [], {}
    for variant, rate in itertools.product(VARIANTS, args.rates):
        runs = [train(variant, rate, seed, data["train"], data["val"], device, args.epochs, args.batch_size) for seed in SEEDS]
        sweep.append({"variant": variant, "learning_rate": rate, "val_accuracy": float(np.mean([a for _, a in runs]))})
        models[(variant, rate)] = [m for m, _ in runs]
        print(sweep[-1], flush=True)
    winners = {v: max((e for e in sweep if e["variant"] == v), key=lambda e: e["val_accuracy"]) for v in VARIANTS}
    chosen_models = {v: models[(v, winners[v]["learning_rate"])] for v in VARIANTS}
    first, second = first_rankers(device), v2_rankers(device)

    result = {"questions": {s: len(q) for s, q in data.items()}, "sweep": sweep, "winners": winners,
              "source_accuracy": {s: source_accuracy(data[s]) for s in ("val", "test")}, "val": {}, "test": {}}
    configs = [("1", k, name) for k in (0, 5) for name in SETS if not (k == 0 and name == "none")]
    per_question = {"val": {}, "test": {}}
    for split in ("val", "test"):
        for temperature, k, name in configs:
            config = (temperature, k, SETS[name])
            key = f"T{temperature}_k{k}_{name}"
            outcomes = untrained(data[split], config)
            outcomes["ranker_v1"] = choose_v2(first, data[split], config, device, first_ranker=True)
            for variant, ms in second.items():
                outcomes[f"ranker_v2_{variant}"] = choose_v2(ms, data[split], config, device)
            for variant, ms in chosen_models.items():
                outcomes[f"ranker_v3_{variant}"] = choose(ms, data[split], config, device)
            result[split][key] = rows_for(data[split], outcomes)
            for selector, chosen in outcomes.items():
                if selector != "oracle":
                    per_question[split][(key, selector)] = chosen
        print(f"[evaluated] {split}", flush=True)

    choices, routed = router(data, per_question)
    greedy = {s: per_question[s][("T1_k0_cot", "greedy")] for s in ("val", "test")}
    result["router"] = {"choices": choices,
                        **{s: rows_for(data[s], {"greedy": greedy[s], "router": routed[s]})["router"] for s in ("val", "test")}}

    candidates = [(key, sel) for key in result["val"] for sel in result["val"][key] if sel.startswith("ranker")]
    best_key, best_sel = max(candidates, key=lambda c: result["val"][c[0]][c[1]]["pooled"]["accuracy"])
    result["choice"] = {"configuration": best_key, "selector": best_sel, "val": result["val"][best_key][best_sel]["pooled"],
                        "test": result["test"][best_key]}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    torch.save({v: [m.state_dict() for m in ms] for v, ms in chosen_models.items()}, RESULTS / "candidate_ranker_v3_models.pt")
    args.output.with_suffix(".md").write_text(markdown(result, sorted(second)), encoding="utf-8")
    print(f"[done] wrote {args.output}")


def markdown(result: dict, v2_variants: list[str]) -> str:
    """Chosen ranker per dataset, the router, then every configuration on test."""
    c = result["choice"]
    test = c["test"]
    lines = ["# Candidate ranker v3: reasoning and crops in one ranker, with source and dataset features", "",
             f"Chosen on validation: `{c['configuration']}` with `{c['selector']}` (validation gain {100 * c['val']['difference']:+.2f} points). "
             f"Questions: {result['questions']}. Baseline to beat: v2 `T1_k0_cot` scalars_judge, +1.84 on test.", "",
             "| Dataset | Greedy | Chosen | Difference (95% interval) | Fixed / broken | p | Oracle |", "|---|---:|---:|---|---|---:|---:|"]
    for d in ("pooled", *DATASET_NAMES):
        r = test[c["selector"]][d]
        lines.append(f"| {d} | {100 * r['greedy']:.2f}% | {100 * r['accuracy']:.2f}% | {100 * r['difference']:+.2f} ({100 * r['ci_low']:+.2f} to {100 * r['ci_high']:+.2f}) | "
                     f"+{r['fixed']} / -{r['broken']} | {r['p_value']:.4f} | {100 * test['oracle'][d]['accuracy']:.2f}% |")
    lines += ["", "## Per-dataset router (chosen on validation, untrained)", "",
              "| Dataset | Configuration | Selector | Val accuracy | Test difference (95% interval) |", "|---|---|---|---:|---|"]
    router_test = result["router"]["test"]
    for d in DATASET_NAMES:
        ch = result["router"]["choices"].get(d)
        if ch:
            r = router_test[d]
            lines.append(f"| {d} | `{ch['configuration']}` | `{ch['selector']}` | {100 * ch['val_accuracy']:.2f}% | "
                         f"{100 * r['difference']:+.2f} ({100 * r['ci_low']:+.2f} to {100 * r['ci_high']:+.2f}) |")
    r = router_test["pooled"]
    lines.append(f"| pooled | | | | {100 * r['difference']:+.2f} ({100 * r['ci_low']:+.2f} to {100 * r['ci_high']:+.2f}), accuracy {100 * r['accuracy']:.2f}% |")
    names = (["greedy", "head_best", "judge_best", "ranker_v1"] + [f"ranker_v2_{v}" for v in v2_variants]
             + [f"ranker_v3_{v}" for v in VARIANTS] + ["oracle"])
    lines += ["", "## Every configuration on test (pooled accuracy)", "", "| Configuration | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key, rows in result["test"].items():
        lines.append(f"| {key} | " + " | ".join(f"{100 * rows[n]['pooled']['accuracy']:.2f}%" for n in names) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
