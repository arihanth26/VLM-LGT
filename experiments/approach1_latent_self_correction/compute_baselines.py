# compute_baselines.py
#
# Training-free baselines for the correction step, on the same questions and candidates as the v3
# ranker, with the same cost accounting, so every method can be placed on one accuracy-against-cost
# axis. Nothing here is trained or tuned on test: the baselines have no parameters except the number of
# answers they vote over, and every setting is reported (none is picked).
#
# Baselines (each question's final answer):
#   greedy              the 2B greedy answer (1 generation call)
#   self_consistency_k  majority vote over the greedy answer and the first k samples at T = 1.0
#                       (1 + k calls); ties go to the earliest answer, so to greedy first
#   reasoning_vote      majority vote over the reasoning answer and the 3 sampled reasoning answers
#                       (1 + 4 calls; the greedy answer is the fallback when none has an "Answer:" line)
#   reasoning_greedy_vote   the same vote with the greedy answer added
#   set_vote / set_judge_best / set_token_best
#                       on the ranker's best candidate set (greedy, grounded crops, reasoning, sampled
#                       reasoning): majority vote, the model's own yes/no judge, and the highest answer
#                       probability. The same candidates the ranker sees, without the ranker.
#   model_4b / model_8b the greedy answer of a larger Qwen3-VL (escalation.py), for every question
# Cost: generation calls as in train_ranker_v3.generation_calls (a 4B answer counts 2, an 8B answer 4,
# by parameter count), and passes = calls + 2 scoring passes per distinct candidate for the methods
# that score candidates (judge and token best); voting needs no scoring pass.
# Votes compare answers after normalize_vqa_answer, the normalization the ranker's agreement features use.
#
# Writes results/baselines.json and results/baselines.md (test, with paired comparisons against greedy).

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.approach1_latent_self_correction.correction_scale import ALL_DATASETS, DATASET_NAMES
from experiments.approach1_latent_self_correction.correction_stats import policy_row
from experiments.approach1_latent_self_correction.scoring import normalize_vqa_answer
from experiments.approach1_latent_self_correction.train_candidate_ranker import load_questions
from experiments.approach1_latent_self_correction.train_ranker_v2 import RESULTS, merge_v2
from experiments.approach1_latent_self_correction.train_ranker_v3 import COT_SAMPLES, MULTI_SETS, merge_cot_samples

ESCALATION = RESULTS / "escalation"
MODEL_CALLS = {"4b": 2.0, "8b": 4.0}
SC_SAMPLES = (4, 9)
BEST_SET = MULTI_SETS["ground_cot_multi"]


def vote(answers: list[str]) -> str:
    """The most common answer after normalization; ties go to the earliest answer."""
    counts = Counter(normalize_vqa_answer(a) for a in answers)
    top = max(counts.values())
    return next(a for a in answers if counts[normalize_vqa_answer(a)] == top)


def correct(q: dict, answer: str) -> bool:
    """Whether an answer string is right; every candidate the pipeline produced is encoded with its label."""
    return bool(q["encoded"][answer]["correct"])


def candidates(q: dict, sources: tuple) -> list[str]:
    """Greedy first, then the answers of the given correction sources that exist for this question."""
    return [q["greedy"]] + [q["corrections"][s] for s in sources if s in q["corrections"]]


def methods(q: dict, larger: dict) -> dict[str, tuple[bool, float, float]]:
    """(correct, generation calls, passes) of every baseline for one question."""
    out = {"greedy": (correct(q, q["greedy"]), 1.0, 1.0)}
    for k in SC_SAMPLES:
        answers = [q["greedy"]] + q["samples"]["1"][:k]
        out[f"self_consistency_{k + 1}"] = (correct(q, vote(answers)), 1.0 + k, 1.0 + k)
    reasoning = [q["corrections"][s] for s in ("cot",) + COT_SAMPLES if s in q["corrections"]]
    out["reasoning_vote"] = (correct(q, vote(reasoning)) if reasoning else correct(q, q["greedy"]), 5.0, 5.0)
    out["reasoning_greedy_vote"] = (correct(q, vote([q["greedy"]] + reasoning)), 5.0, 5.0)
    pool = candidates(q, BEST_SET)
    distinct = list(dict.fromkeys(pool))
    calls = 8.0  # greedy + box + 2 grounded crops + reasoning + 3 sampled reasoning answers
    scored = calls + 2 * len(distinct)
    out["set_vote"] = (correct(q, vote(pool)), calls, calls)
    out["set_judge_best"] = (correct(q, max(distinct, key=lambda a: (q["judge"].get(a, 0.0), a == q["greedy"]))), calls, scored)
    out["set_token_best"] = (correct(q, max(distinct, key=lambda a: (q["encoded"][a]["tf_confidence"], a == q["greedy"]))), calls, scored)
    for size, answers in larger.items():
        if q["key"] in answers:
            out[f"model_{size}"] = (bool(answers[q["key"]]["correct"]), MODEL_CALLS[size], MODEL_CALLS[size])
    return out


def larger_models(split: str) -> dict[str, dict]:
    """Greedy answers of the larger models, by (dataset, index), for whichever sizes escalation.py produced."""
    out = {}
    for folder in sorted(ESCALATION.glob("model_*")):
        rows = {}
        for path in sorted((folder / split).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line) if line.strip() else {}
                if "answer" in r:
                    rows[(r["dataset"], r["index"])] = r["answer"]
        if rows:
            out[folder.name.split("_", 1)[1]] = rows
    return out


def evaluate(questions: list[dict], larger: dict) -> dict:
    """Accuracy, cost and the paired comparison with greedy for every baseline, pooled and per dataset."""
    rows = [methods(q, larger) for q in questions]
    names = [n for n in rows[0] if all(n in r for r in rows)]
    datasets = np.array([q["key"][0] for q in questions])
    greedy = np.array([r["greedy"][0] for r in rows])
    out = {}
    for name in names:
        final = np.array([r[name][0] for r in rows])
        entry = {"calls": float(np.mean([r[name][1] for r in rows])), "passes": float(np.mean([r[name][2] for r in rows])),
                 "pooled": policy_row(greedy.tolist(), final.tolist())}
        for d in ALL_DATASETS:  # whichever datasets the split contains (the unseen split has only its own)
            m = datasets == d
            if m.any():
                entry[d] = policy_row(greedy[m].tolist(), final[m].tolist())
        out[name] = entry
    return out


def markdown(result: dict) -> str:
    """Test table, sorted by cost."""
    pct = lambda x: f"{100 * x:.2f}%"
    pts = lambda x: f"{100 * x:+.2f}"
    lines = ["# Training-free baselines on the correction candidates", "",
             "Same questions and candidates as the v3 ranker; nothing is selected on test. Calls: generation calls per question "
             "(a 4B answer counts 2, an 8B answer 4). Passes add the scoring passes of methods that score candidates.", "",
             "| Method | Calls | Passes | Test accuracy | Gain over greedy (95% interval) | Fixed / broken | " + " | ".join(DATASET_NAMES) + " |",
             "|---|---:|---:|---:|---|---|" + "---|" * len(DATASET_NAMES)]
    for name, e in sorted(result["test"].items(), key=lambda kv: (kv[1]["calls"], kv[0])):
        p = e["pooled"]
        lines.append(f"| {name} | {e['calls']:.1f} | {e['passes']:.1f} | {pct(p['accuracy'])} | {pts(p['difference'])} ({pts(p['ci_low'])} to {pts(p['ci_high'])}) | "
                     f"+{p['fixed']} / -{p['broken']} | " + " | ".join(pts(e[d]["difference"]) for d in DATASET_NAMES) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Load the candidates with the sampled reasoning answers, evaluate every baseline on val and test."""
    parser = argparse.ArgumentParser(description="Training-free baselines on the correction candidates.")
    parser.add_argument("--output", type=Path, default=RESULTS / "baselines.json")
    args = parser.parse_args()
    result = {}
    for split in ("val", "test"):
        questions = merge_v2(load_questions(split), split)
        merge_cot_samples(questions, split)
        result[split] = evaluate(questions, larger_models(split))
        print(f"[evaluated] {split}: {len(questions)} questions, {len(result[split])} methods", flush=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"[done] wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
