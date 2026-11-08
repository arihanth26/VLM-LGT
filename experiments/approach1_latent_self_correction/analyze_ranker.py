# analyze_ranker.py
#
# Analysis behind report 11 (the learned correction step). Loads the rankers trained by
# train_candidate_ranker.py (three seeds per variant, scores averaged) and the validation and test
# candidates, including the samples past ten from extend_pools.py, and measures:
#   k curve        accuracy against the number of samples k = 0..16 at T = 1.0 and 1.3, with no
#                  corrections, zoomed crops, and all corrections, for greedy, head best, vote,
#                  the ranker, and the oracle, pooled and per dataset, with paired tests
#   within-question ranking   grouped AUROC and mixed-pool accuracy of the ranker against the head
#   attribution    which strategy proposed the answer the ranker chose when it changed the greedy
#                  answer, and how many of those changes fixed or broke it
#   gating         accuracy when only the lowest-scored share of questions is corrected, with the
#                  number of model passes per question
#   cases          examples the ranker fixed and broke, with thumbnails
# Writes results/ranker_analysis.json. Runs on a GPU node.

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.candidate_pools import cached_split
from experiments.approach1_latent_self_correction.baseline import extract_image
from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES, dataset_dir
from experiments.approach1_latent_self_correction.correction_stats import paired_difference
from experiments.approach1_latent_self_correction.train_candidate_ranker import (
    CORRECTION_SETS,
    Ranker,
    batch_tensors,
    candidate_set,
    load_questions,
    untrained_selectors,
)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
K_VALUES = (0, 1, 3, 5, 7, 9, 10, 11, 13, 15, 16)
CURVE_SETS = ("none", "zoom", "all")
GATES = ("head", "token", "pre_head")
BUDGETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def load_rankers(device) -> dict[str, list[Ranker]]:
    """The three seeds of each trained variant."""
    winners = json.loads((RESULTS / "candidate_ranker.json").read_text())["winners"]
    states = torch.load(RESULTS / "candidate_ranker_models.pt", map_location=device, weights_only=False)
    rankers = {}
    for variant, seeds in states.items():
        rankers[variant] = []
        for state in seeds:
            model = Ranker(variant, winners[variant]["hidden"]).to(device)
            model.load_state_dict(state)
            model.eval()
            rankers[variant].append(model)
    return rankers


def score_sets(models: list[Ranker], questions: list[dict], config: tuple, device, batch_size: int = 256) -> list[tuple]:
    """Per question: candidate rows, features, labels, and the seed-averaged ranker scores."""
    temperature, k, sources = config
    result = []
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            part = questions[start:start + batch_size]
            sets = [candidate_set(q, temperature, k, sources) for q in part]
            batch = batch_tensors(part, sets, device)
            scores = torch.stack([m(batch["question"], batch["answer"], batch["scalars"]) for m in models]).mean(0).cpu().numpy()
            for (rows, features, labels), row_scores in zip(sets, scores):
                result.append((rows, features, labels, row_scores[:len(rows)]))
    return result


def chosen_correct(scored: list[tuple]) -> list[bool]:
    """Whether the top-scored candidate is correct, per question."""
    return [bool(labels[int(np.argmax(s))]) for _, _, labels, s in scored]


def grouped(scored: list[tuple], column: int | None) -> dict:
    """Within-question AUROC and mixed-pool accuracy of the ranker (column None) or a scalar feature."""
    aucs, picks, random_picks = [], [], []
    for _, features, labels, scores in scored:
        values = scores if column is None else features[:, column]
        right, wrong = values[labels == 1], values[labels == 0]
        if len(right) and len(wrong):
            comparison = right[:, None] - wrong[None, :]
            aucs.append((comparison > 0).mean() + 0.5 * (comparison == 0).mean())
            picks.append(bool(labels[int(np.argmax(values))]))
            random_picks.append(float(labels.mean()))
    return {"grouped_auroc": float(np.mean(aucs)), "mixed_accuracy": float(np.mean(picks)),
            "mixed_random_pick": float(np.mean(random_picks)), "mixed_questions": len(aucs)}


def by_dataset(questions: list[dict], greedy: list[bool], chosen: list[bool]) -> dict:
    """Pooled and per-dataset accuracy and paired difference from greedy."""
    out = {"pooled": {**paired_difference(greedy, chosen), "accuracy": float(np.mean(chosen)), "greedy": float(np.mean(greedy))}}
    for name in DATASET_NAMES:
        mask = [q["key"][0] == name for q in questions]
        g = [x for x, m in zip(greedy, mask) if m]
        c = [x for x, m in zip(chosen, mask) if m]
        out[name] = {**paired_difference(g, c), "accuracy": float(np.mean(c)), "greedy": float(np.mean(g))}
    return out


def k_curve(questions: list[dict], ranker: list[Ranker], device) -> dict:
    """Accuracy of every selector at every k, temperature and correction set."""
    curve = {}
    longest = {t: min(len(q["samples"].get(t, [])) for q in questions) for t in ("1", "1.3")}
    curve["longest_pool"] = longest
    print(f"[curve] samples available for every question: {longest}", flush=True)
    for temperature in ("1", "1.3"):
        for name in CURVE_SETS:
            for k in K_VALUES:
                if k > longest[temperature] or (k == 0 and name == "none"):
                    continue
                config = (temperature, k, CORRECTION_SETS[name])
                untrained = untrained_selectors(questions, config)
                scored = score_sets(ranker, questions, config, device)
                greedy = untrained["greedy"]
                entry = {selector: by_dataset(questions, greedy, outcome) for selector, outcome in untrained.items()}
                entry["ranker"] = by_dataset(questions, greedy, chosen_correct(scored))
                entry["within"] = {"ranker": grouped(scored, None), "head": grouped(scored, 6), "vote": grouped(scored, 0)}
                curve[f"T{temperature}_k{k}_{name}"] = entry
                print(f"[curve] T{temperature} k{k} {name}: ranker {entry['ranker']['pooled']['accuracy']:.4f}", flush=True)
    return curve


def proposers(question: dict, answer: str, temperature: str, k: int, sources: tuple) -> list[str]:
    """Which candidate sources proposed an answer: greedy, samples, contrastive decoding, zoomed crops."""
    found = []
    if answer == question["greedy"]:
        found.append("greedy")
    if answer in question["samples"][temperature][:k]:
        found.append("sample")
    for source in sources:
        if question["corrections"].get(source) == answer:
            found.append(source.split("_")[0])
    return sorted(set(found))


def attribution(questions: list[dict], scored: list[tuple], config: tuple) -> dict:
    """Changes from the greedy answer, split by which sources proposed the chosen answer."""
    temperature, k, sources = config
    table: dict[str, dict] = {}
    for question, (rows, features, labels, scores) in zip(questions, scored):
        top = int(np.argmax(scores))
        if features[top, 1] == 1.0:
            continue
        answer = question["answers"][rows[top]]
        greedy_correct = bool(labels[int(np.argmax(features[:, 1]))])
        key = "+".join(proposers(question, answer, temperature, k, sources)) or "other"
        entry = table.setdefault(key, {"changed": 0, "fixed": 0, "broken": 0, "wrong_to_wrong": 0, "right_to_right": 0})
        entry["changed"] += 1
        if labels[top] and not greedy_correct:
            entry["fixed"] += 1
        elif greedy_correct and not labels[top]:
            entry["broken"] += 1
        elif labels[top]:
            entry["right_to_right"] += 1
        else:
            entry["wrong_to_wrong"] += 1
    return dict(sorted(table.items(), key=lambda item: -item[1]["changed"]))


def gate_values(question: dict, gate: str) -> float:
    """Question-level gate: the head or token score of the greedy answer, or the pre-generation score."""
    greedy = question["encoded"][question["greedy"]]
    if gate == "head":
        return greedy["head_confidence"]
    if gate == "token":
        return greedy["tf_confidence"]
    return question["pre_generation"]


def gating(questions: list[dict], scored: list[tuple], config: tuple) -> dict:
    """Correct only the lowest-gated share of questions; the rest keep the greedy answer."""
    temperature, k, sources = config
    greedy = np.array([bool(labels[int(np.argmax(features[:, 1]))]) for _, features, labels, _ in scored])
    chosen = np.array(chosen_correct(scored))
    # A corrected question costs one batched sample call (if k > 0), one call per correction
    # (contrastive decoding runs two forward passes per token), and one encoding pass per distinct
    # candidate; an uncorrected one costs the greedy generation only.
    cost = np.array([1 + (1 if k else 0) + sum(2 if s.startswith("vcd") else 1 for s in sources) + len(rows)
                     for rows, _, _, _ in scored])
    result = {}
    for gate in GATES:
        values = np.array([gate_values(q, gate) for q in questions])
        order = np.argsort(values, kind="stable")
        rows = {}
        for budget in BUDGETS:
            routed = np.zeros(len(questions), dtype=bool)
            routed[order[: int(round(budget * len(questions)))]] = True
            final = np.where(routed, chosen, greedy)
            rows[f"{budget:.1f}"] = {**paired_difference(greedy.tolist(), final.tolist()), "accuracy": float(final.mean()),
                                     "passes_per_question": float(np.where(routed, cost, 1).mean())}
        result[gate] = rows
    return result


def thumbnail(dataset: str, split: str, index: int, cache: dict) -> str:
    """A small JPEG of the question's image, base64-encoded for the report."""
    if dataset not in cache:
        cache[dataset] = cached_split(dataset, split)
    image = extract_image(cache[dataset][index], dataset_dir(dataset)).convert("RGB")
    image.thumbnail((360, 360))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def cases(questions: list[dict], scored: list[tuple], config: tuple, per_dataset: int = 2) -> list[dict]:
    """Fixed and broken examples per dataset, with every candidate, its sources, scores and correctness."""
    temperature, k, sources = config
    picked, cache = [], {}
    for name in DATASET_NAMES:
        counts = {"fixed": 0, "broken": 0}
        for question, (rows, features, labels, scores) in zip(questions, scored):
            if question["key"][0] != name:
                continue
            top, greedy_row = int(np.argmax(scores)), int(np.argmax(features[:, 1]))
            kind = "fixed" if labels[top] and not labels[greedy_row] else "broken" if labels[greedy_row] and not labels[top] else None
            limit = per_dataset if kind == "fixed" else 1
            if kind is None or counts[kind] >= limit:
                continue
            counts[kind] += 1
            candidates = [{"answer": question["answers"][r], "correct": bool(labels[i]), "ranker": float(scores[i]),
                           "head": float(features[i, 6]), "vote": float(features[i, 0]),
                           "sources": proposers(question, question["answers"][r], temperature, k, sources)}
                          for i, r in enumerate(rows)]
            picked.append({"dataset": name, "index": question["key"][1], "kind": kind, "question": question["question_text"],
                           "reference": question["reference"], "greedy": question["greedy"],
                           "chosen": question["answers"][rows[top]], "candidates": sorted(candidates, key=lambda c: -c["ranker"]),
                           "image": thumbnail(name, "test", question["key"][1], cache)})
    return picked


def attach_text(questions: list[dict], split: str) -> None:
    """Question text, reference answer and pre-generation score from the pools, for cases and gating."""
    pools = {}
    for path in sorted((HERE / "results" / "pools" / "T1" / split).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if "greedy" in record:
                    pools[(record["dataset"], record["index"])] = record
    for question in questions:
        record = pools[question["key"]]
        question.update(question_text=record.get("question", ""), reference=record.get("reference_answer", ""),
                        pre_generation=record["pre_generation"]["head"])


def main() -> None:
    """Run every analysis and write results/ranker_analysis.json."""
    parser = argparse.ArgumentParser(description="Analysis of the learned candidate ranker for report 11.")
    parser.add_argument("--output", type=Path, default=RESULTS / "ranker_analysis.json")
    parser.add_argument("--limit", type=int, default=None, help="Every n-th question only, for a smoke test.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rankers = load_rankers(device)
    result = {"questions": {}, "choice": json.loads((RESULTS / "candidate_ranker.json").read_text())["choice"]["configuration"]}
    for split in ("val", "test"):
        questions = load_questions(split, include_extra=True)
        if args.limit:
            questions = questions[:: max(1, len(questions) // args.limit)]
        attach_text(questions, split)
        result["questions"][split] = len(questions)
        result[f"{split}_curve"] = k_curve(questions, rankers["full"], device)
        if split == "test":
            main_config = ("1", 5, CORRECTION_SETS["all"])
            zoom_config = ("1", 0, CORRECTION_SETS["zoom"])
            for label, config in (("main", main_config), ("zoom_only", zoom_config)):
                scored = score_sets(rankers["full"], questions, config, device)
                result[f"attribution_{label}"] = attribution(questions, scored, config)
                result[f"gating_{label}"] = gating(questions, scored, config)
            result["variants"] = {variant: by_dataset(questions, untrained_selectors(questions, main_config)["greedy"],
                                                      chosen_correct(score_sets(models, questions, main_config, device)))
                                  for variant, models in rankers.items()}
            result["cases"] = cases(questions, score_sets(rankers["full"], questions, main_config, device), main_config)
        print(f"[done] {split}", flush=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
