# analyze_best_ranker.py
#
# Evidence for the best correction step (ranker v3 with the 8B short answer and 8B step-by-step answer as
# candidates, trained without the judge input; candidate_ranker_v3_multi_esc_cot8b_drop_judge_models.pt), on the
# test split, with the saved models exactly as trained (no retraining, nothing chosen on test):
#
#   paired      the best ranker (no gate, and gated at 50% with the validation threshold of the answer probability)
#               against the alternatives on the same questions: routing to 8B with the setting chosen on validation
#               in escalation.json, the 8B answer on every question, and the 2B-only ranker without the judge
#               (candidate_ranker_v3_multi_drop_judge_models.pt, greedy + grounded crops + reasoning + 3 samples).
#               Paired bootstrap interval and McNemar p for each difference, pooled and per dataset.
#   origin      where the chosen answer comes from: the greedy answer kept, an answer only 2B corrections proposed,
#               one only the 8B model proposed, or one both proposed; how often each is chosen, how often it is right,
#               and the wrong greedy answers it fixes and the right ones it breaks.
#   oracle      how often a right answer is among the candidates, and how often the ranker then picks one.
#   selective   answering only the questions the system is most confident about: accuracy at 50/80/90/100% coverage
#               and the area under the risk-coverage curve, for the ranker's answer ranked by the ranker's own
#               probability, against the greedy answer ranked by its answer probability.
#   examples    a few fixed and broken questions per dataset (question, reference, greedy and chosen answers, origin).
#
# With another base model (VLMLGT_MODEL=4b) every path is under results/base_4b/; the 2B-only ranker there is the one
# trained with the judge (--baseline-models candidate_ranker_v3_multi_models.pt --baseline-with-judge).
# Writes results/best_ranker_analysis.json and .md (results/base_<size>/ for another base model).

import argparse
import json

import numpy as np
import torch

from experiments.approach1_latent_self_correction import train_ranker_v3 as v3
from experiments.approach1_latent_self_correction.analyze_escalation import final as route_final
from experiments.approach1_latent_self_correction.analyze_escalation import load_fixes, load_greedy, routed, table
from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.correction_stats import paired_difference, policy_row
from experiments.approach1_latent_self_correction.gate_ranker_v3 import load_models
from experiments.approach1_latent_self_correction.gating import combine, gate, threshold
from experiments.approach1_latent_self_correction.train_candidate_ranker import load_questions
from experiments.approach1_latent_self_correction.train_ranker_v2 import RESULTS, merge_v2

BEST_MODELS = RESULTS / "candidate_ranker_v3_multi_esc_cot8b_drop_judge_models.pt"
TWO_B_MODELS = RESULTS / "candidate_ranker_v3_multi_drop_judge_models.pt"
BEST_SET, TWO_B_SET = "ground_cot_multi_esc8b_cot8b", "ground_cot_multi"
VARIANT = "scalars_ds"
LARGER = ("esc_8b", "cot_8b")
COVERAGES = (0.5, 0.8, 0.9, 1.0)
EXAMPLES_PER_DATASET = 3


def load(split: str) -> list[dict]:
    """Questions with every candidate source merged: rounds 1-3, the 8B answers and the 8B reasoning."""
    questions = merge_v2(load_questions(split), split)
    v3.merge_cot_samples(questions, split)
    v3.merge_cot_samples(questions, split, v3.V4_DIR)
    v3.merge_cot_samples(questions, split, v3.V5_DIR)
    return questions


def pools(split: str) -> dict:
    """The T=1 pool record of every question (greedy answer probability, question text, reference)."""
    out = {}
    for path in sorted((RESULTS / "pools" / "T1" / split).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if "greedy" in r:
                out[(r["dataset"], r["index"])] = r
    return out


def pick(models: list, questions: list[dict], sources: tuple, device, batch_size: int = 256) -> list[dict]:
    """Per question: the chosen answer, whether it is right, its probability under the ranker, and whether any
    candidate is right (the same seed-averaged scores as train_ranker_v3.choose)."""
    out = []
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            part = questions[start:start + batch_size]
            sets = [v3.candidate_set(q, "1", 0, sources) for q in part]
            batch = v3.batch_tensors(part, sets, device)
            scores = torch.stack([m(batch["question"], batch["answer"], batch["scalars"]) for m in models]).mean(0)
            scores = scores.masked_fill(~batch["mask"], float("-inf"))
            probs = torch.softmax(scores, dim=1).cpu().numpy()
            top = scores.argmax(1).cpu().numpy()
            for q, (rows, _, labels), i, p in zip(part, sets, top, probs):
                out.append({"answer": q["answers"][rows[i]], "correct": bool(labels[i]), "probability": float(p[i]),
                            "any_correct": bool(labels.max() > 0)})
    return out


def origin(q: dict, answer: str, sources: tuple) -> str:
    """Which family proposed the chosen answer."""
    if answer == q["greedy"]:
        return "greedy kept"
    proposed = {s for s in sources if q["corrections"].get(s) == answer}
    larger, smaller = bool(proposed & set(LARGER)), bool(proposed - set(LARGER))
    return "2B and 8B both" if larger and smaller else "8B only" if larger else "2B corrections only" if smaller else "2B sample"


def selective(correct: np.ndarray, confidence: np.ndarray) -> dict:
    """Accuracy when answering only the most confident share of questions, and the area under the risk-coverage curve."""
    order = np.argsort(-confidence, kind="stable")
    ranked = correct[order].astype(float)
    running = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    out = {f"accuracy_at_{int(c * 100)}": float(running[max(0, int(round(c * len(ranked))) - 1)]) for c in COVERAGES}
    out["aurc"] = float(np.mean(1 - running))  # lower is better
    return out


def paired_rows(datasets: np.ndarray, base: np.ndarray, other: np.ndarray) -> dict:
    """Paired difference other - base, pooled and per dataset."""
    rows = {"pooled": paired_difference(base.tolist(), other.tolist())}
    for d in DATASET_NAMES:
        m = datasets == d
        rows[d] = paired_difference(base[m].tolist(), other[m].tolist())
    return rows


def routing_outcome(keys: list[tuple]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-question correctness of routing to 8B with the validation-chosen setting, and of 8B everywhere."""
    chosen = json.loads((RESULTS / "escalation.json").read_text())["fixes"]["8b"]
    best = next(h for h in chosen["headlines"] if h["name"] == "best detector, rule and budget")
    val = table(load_greedy("val"), load_fixes("val")["8b"])
    greedy_test, fix_test = load_greedy("test"), load_fixes("test")["8b"]
    test = table(greedy_test, fix_test)
    final = route_final(test, routed(test, best["detector"], best["budget"], val), best["rule"])
    position = {k: i for i, k in enumerate(sorted(greedy_test))}
    route = np.array([final[position[k]] for k in keys])
    everywhere = np.array([test["fix"][position[k]] for k in keys])
    return route, everywhere, {k: best[k] for k in ("detector", "rule", "budget")}


def markdown(r: dict) -> str:
    """Tables of the analysis."""
    pct = lambda x: f"{100 * x:.2f}%"
    ci = lambda d: f"{100 * d['difference']:+.2f} ({100 * d['ci_low']:+.2f} to {100 * d['ci_high']:+.2f}), p = {d['p_value']:.2g}"
    lines = ["# The best ranker: paired comparisons, where its answers come from, abstention", "",
             f"Test, {r['questions']} questions. Best ranker: {BEST_SET}, no judge input. Routing setting chosen on validation: {r['routing_setting']}.", "",
             "## Paired comparisons (row minus column, same questions)", "", "| Comparison | Pooled | " + " | ".join(DATASET_NAMES) + " |",
             "|---|---|" + "---|" * len(DATASET_NAMES)]
    for name, rows in r["paired"].items():
        lines.append(f"| {name} | {ci(rows['pooled'])} | " + " | ".join(f"{100 * rows[d]['difference']:+.2f}" for d in DATASET_NAMES) + " |")
    lines += ["", "## Accuracy of each method", "", "| Method | Test accuracy |", "|---|---:|"]
    lines += [f"| {k} | {pct(v)} |" for k, v in r["accuracy"].items()]
    lines += ["", "## Where the chosen answer comes from (no gate)", "", "| Origin | Chosen | Right when chosen | Fixed | Broken |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {k} | {v['share'] * 100:.1f}% | {pct(v['accuracy'])} | {v['fixed']} | {v['broken']} |" for k, v in r["origin"].items()]
    o = r["oracle"]
    lines += ["", f"Oracle: a right answer is among the candidates for {pct(o['any_correct'])} of questions; the ranker picks a right one in "
              f"{pct(o['picked_when_available'])} of those; {o['missed']} questions have a right candidate the ranker misses.", "",
              "## Answering only the most confident questions", "", "| Answers ranked by | " + " | ".join(f"accuracy at {int(c * 100)}%" for c in COVERAGES) + " | AURC |",
              "|---|" + "---:|" * (len(COVERAGES) + 1)]
    for k, v in r["selective"].items():
        lines.append(f"| {k} | " + " | ".join(pct(v[f'accuracy_at_{int(c * 100)}']) for c in COVERAGES) + f" | {v['aurc']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Score both rankers on test, compare with routing, write the analysis."""
    parser = argparse.ArgumentParser(description="Paired comparisons and analyses of the best ranker.")
    parser.add_argument("--output", default=str(RESULTS / "best_ranker_analysis.json"))
    parser.add_argument("--baseline-models", default=str(TWO_B_MODELS), help="The ranker without the 8B candidates.")
    parser.add_argument("--baseline-with-judge", action="store_true", help="That ranker was trained with the judge input.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3.SETS.update(v3.MULTI_SETS)
    if not args.baseline_with_judge:
        v3.DROPPED.update(v3.FEATURE_GROUPS["judge"])  # the baseline ranker was trained without the judge
    questions = load("test")
    val_tokens = np.array([r["greedy"]["token_confidence"] for r in pools("val").values()])
    test_pools = pools("test")
    keys = [q["key"] for q in questions]
    datasets = np.array([k[0] for k in keys])
    greedy = np.array([bool(q["encoded"][q["greedy"]]["correct"]) for q in questions])
    tokens = np.array([test_pools[k]["greedy"]["token_confidence"] for k in keys])

    # Phase 1, the 2B-only ranker: trained without the escalation group, so it is scored before that group is enabled.
    two_b = pick(load_models(args.baseline_models, VARIANT, device), questions, v3.SETS[TWO_B_SET], device)
    v3.DROPPED.update(v3.FEATURE_GROUPS["judge"])  # the best ranker is always the one trained without the judge
    v3.enable_escalation()
    best = pick(load_models(BEST_MODELS, VARIANT, device), questions, v3.SETS[BEST_SET], device)
    sources = v3.SETS[BEST_SET]

    best_c = np.array([b["correct"] for b in best])
    two_c = np.array([b["correct"] for b in two_b])
    gated = combine(gate(tokens, threshold(val_tokens, 0.5)), best_c, greedy)
    route, everywhere, setting = routing_outcome(keys)

    origins = [origin(q, b["answer"], sources) for q, b in zip(questions, best)]
    origin_rows = {}
    for name in ("greedy kept", "2B corrections only", "8B only", "2B and 8B both", "2B sample"):
        m = np.array([o == name for o in origins])
        if m.any():
            origin_rows[name] = {"share": float(m.mean()), "accuracy": float(best_c[m].mean()),
                                 "fixed": int((m & ~greedy & best_c).sum()), "broken": int((m & greedy & ~best_c).sum())}
    any_c = np.array([b["any_correct"] for b in best])
    probability = np.array([b["probability"] for b in best])

    examples = []
    for d in DATASET_NAMES:
        for kind, mask in (("fixed", ~greedy & best_c), ("broken", greedy & ~best_c)):
            for i in np.flatnonzero(mask & (datasets == d))[:EXAMPLES_PER_DATASET if kind == "fixed" else 1]:
                q, p = questions[i], test_pools[keys[i]]
                examples.append({"dataset": d, "index": int(keys[i][1]), "kind": kind, "question": p.get("question"),
                                 "reference": p.get("reference_answer"), "greedy": q["greedy"], "chosen": best[i]["answer"],
                                 "origin": origins[i], "ranker_probability": best[i]["probability"]})

    result = {
        "questions": len(questions), "routing_setting": setting,
        "accuracy": {"2B greedy": float(greedy.mean()), "2B-only ranker (no judge)": float(two_c.mean()),
                     "8B on every question": float(everywhere.mean()), "routing to 8B (validation choice)": float(route.mean()),
                     "best ranker, no gate": float(best_c.mean()), "best ranker, gated at 50%": float(gated.mean())},
        "paired": {"best ranker minus routing to 8B": paired_rows(datasets, route, best_c),
                   "best ranker gated 50% minus routing to 8B": paired_rows(datasets, route, gated),
                   "best ranker minus 8B on every question": paired_rows(datasets, everywhere, best_c),
                   "best ranker minus 2B-only ranker": paired_rows(datasets, two_c, best_c),
                   "routing to 8B minus 8B on every question": paired_rows(datasets, everywhere, route),
                   "2B-only ranker minus 2B greedy": paired_rows(datasets, greedy, two_c)},
        "per_dataset": {d: {"best": policy_row(greedy[datasets == d].tolist(), best_c[datasets == d].tolist())} for d in DATASET_NAMES},
        "origin": origin_rows,
        "oracle": {"any_correct": float(any_c.mean()), "picked_when_available": float(best_c[any_c].mean()), "missed": int((any_c & ~best_c).sum())},
        "selective": {"best ranker's answer, by the ranker's probability": selective(best_c, probability),
                      "best ranker's answer, by the greedy answer's probability": selective(best_c, tokens),
                      "2B greedy answer, by its own probability": selective(greedy, tokens)},
        "examples": examples,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=1) + "\n")
    with open(args.output.replace(".json", ".md"), "w", encoding="utf-8") as f:
        f.write(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
