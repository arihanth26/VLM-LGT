# selection_metrics.py
#
# Measurements for the candidate-search study that target the decision being made, instead of
# pooled correctness AUROC. Pooled AUROC ranks answers of different questions against each other,
# but choosing among candidates only ever compares answers to the same question, and most
# questions give the selector nothing to choose (every candidate right, or every one wrong).
#
# Selection (which candidate to keep):
#   accuracy, and the paired difference from greedy with a bootstrap interval and McNemar p
#   headroom captured      (selector - greedy) / (oracle - greedy)
#   mixed-pool accuracy    accuracy on questions whose pool holds both a right and a wrong answer,
#                          against the random-pick rate on the same questions
#   grouped AUROC          right-versus-wrong ranking within a question, averaged over mixed questions
# Gating (which questions get the extra compute):
#   AURC and E-AURC        area under the risk-coverage curve and its excess over a perfect ranking
#   compute curve          accuracy when the lowest-scored fraction b of questions is sent to a
#                          selector and the rest keep the greedy answer, against average generations
#
# Pool records come from candidate_pools.py. Pool k is the greedy answer plus the first k samples.
# Pure numpy, no model or dataset dependencies.

from collections import defaultdict

import numpy as np

from experiments.approach1_latent_self_correction.correction_stats import paired_difference
from experiments.approach1_latent_self_correction.scoring import normalize_vqa_answer

SCORERS = ("head", "probe", "token")
ANCHOR_MARGINS = (0.0, 0.05, 0.1, 0.2)


def pool(record: dict, k: int) -> list[dict]:
    """The greedy answer followed by the first k sampled answers."""
    return [record["greedy"], *record["samples"][:k]]


def score_of(candidate: dict, scorer: str) -> float:
    """One candidate's score under a scorer."""
    return candidate["token_confidence" if scorer == "token" else f"{scorer}_confidence"]


def argmax(values: list[float]) -> int:
    """First index of the largest value, so ties keep the earlier candidate (greedy first)."""
    return int(np.argmax(np.asarray(values, dtype=float)))


def vote_index(candidates: list[dict], weight: str | None) -> int:
    """Pick the answer group with the most votes (or the largest summed score), then its best-scored member."""
    groups: dict[str, list[int]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        groups[normalize_vqa_answer(candidate["answer"])].append(position)
    total = {key: sum(1.0 if weight is None else score_of(candidates[i], weight) for i in members)
             for key, members in groups.items()}
    best = max(total.values())
    tie_score = weight or "head"
    winners = [members for key, members in groups.items() if total[key] == best]
    members = max(winners, key=lambda group: max(score_of(candidates[i], tie_score) for i in group))
    return max(members, key=lambda i: score_of(candidates[i], tie_score))


def selector_names() -> list[str]:
    """Every selection policy evaluated on a pool."""
    names = ["greedy", "first_sample", "majority_vote", "head_vote", "probe_vote"]
    names += [f"{scorer}_best" for scorer in SCORERS]
    names += [f"head_anchor_{margin:.2f}" for margin in ANCHOR_MARGINS]
    return names + ["oracle"]


def select(record: dict, k: int, selector: str) -> int:
    """Index into pool(record, k) chosen by one selector."""
    candidates = pool(record, k)
    if selector == "greedy":
        return 0
    if selector == "first_sample":
        return 1
    if selector == "majority_vote":
        return vote_index(candidates, None)
    if selector.endswith("_vote"):
        return vote_index(candidates, selector.removesuffix("_vote"))
    if selector.endswith("_best"):
        return argmax([score_of(c, selector.removesuffix("_best")) for c in candidates])
    if selector.startswith("head_anchor_"):
        # Keep greedy unless a sample beats its head score by the margin.
        margin = float(selector.removeprefix("head_anchor_"))
        best = 1 + argmax([c["head_confidence"] for c in candidates[1:]])
        return best if candidates[best]["head_confidence"] > candidates[0]["head_confidence"] + margin else 0
    if selector == "oracle":
        return next((i for i, c in enumerate(candidates) if c["correct"]), 0)
    raise ValueError(f"unknown selector '{selector}'")


def outcomes(records: list[dict], k: int, selector: str) -> list[bool]:
    """Whether each question's selected answer is correct."""
    return [bool(pool(r, k)[select(r, k, selector)]["correct"]) for r in records]


def is_mixed(candidates: list[dict]) -> bool:
    """A pool with at least one right and one wrong answer, the only case where selection matters."""
    flags = {bool(c["correct"]) for c in candidates}
    return flags == {True, False}


def grouped_auroc(records: list[dict], k: int, scorer: str) -> float:
    """Right-versus-wrong ranking inside each mixed pool, averaged over those pools (ties count half)."""
    values = []
    for record in records:
        candidates = pool(record, k)
        if not is_mixed(candidates):
            continue
        right = np.array([score_of(c, scorer) for c in candidates if c["correct"]])
        wrong = np.array([score_of(c, scorer) for c in candidates if not c["correct"]])
        comparison = right[:, None] - wrong[None, :]
        values.append((comparison > 0).mean() + 0.5 * (comparison == 0).mean())
    return float(np.mean(values)) if values else float("nan")


def pool_statistics(records: list[dict], k: int) -> dict:
    """How much the pool offers: oracle rate, mixed pools, distinct answers, format failures, pass rate."""
    candidates = [pool(r, k) for r in records]
    return {
        "questions": len(records),
        "oracle_accuracy": float(np.mean([any(c["correct"] for c in p) for p in candidates])),
        "mixed_fraction": float(np.mean([is_mixed(p) for p in candidates])),
        "mean_distinct_answers": float(np.mean([len({normalize_vqa_answer(c["answer"]) for c in p}) for p in candidates])),
        "unanimous_fraction": float(np.mean([len({normalize_vqa_answer(c["answer"]) for c in p}) == 1 for p in candidates])),
        "sample_pass_rate": float(np.mean([np.mean([c["correct"] for c in r["samples"][:k]]) for r in records])),
        "capped_fraction": float(np.mean([np.mean([c["capped"] for c in r["samples"][:k]]) for r in records])),
    }


def selection_report(records: list[dict], k: int, selector: str, greedy: list[bool], oracle: list[bool]) -> dict:
    """Accuracy, paired gain over greedy, headroom captured, and mixed-pool accuracy of one selector."""
    chosen = outcomes(records, k, selector)
    row = paired_difference(greedy, chosen)
    gap = np.mean(oracle) - np.mean(greedy)
    row.update(accuracy=float(np.mean(chosen)), headroom_captured=float(row["difference"] / gap) if gap > 0 else float("nan"))
    mixed = [i for i, r in enumerate(records) if is_mixed(pool(r, k))]
    if mixed:
        row["mixed_questions"] = len(mixed)
        row["mixed_accuracy"] = float(np.mean([chosen[i] for i in mixed]))
        row["mixed_random_pick"] = float(np.mean([np.mean([c["correct"] for c in pool(records[i], k)]) for i in mixed]))
    return row


def evaluate_pool(records: list[dict], k: int) -> dict:
    """Every selector and every scorer's grouped AUROC for pools of size k."""
    greedy = outcomes(records, k, "greedy")
    oracle = outcomes(records, k, "oracle")
    return {
        "k": k,
        "pool": pool_statistics(records, k),
        "selectors": {name: selection_report(records, k, name, greedy, oracle) for name in selector_names()},
        "grouped_auroc": {scorer: grouped_auroc(records, k, scorer) for scorer in SCORERS},
    }


def aurc(correct: list[bool], scores: list[float]) -> dict:
    """Area under the risk-coverage curve (keep the highest-scored answers first), and its excess over the best ranking."""
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    risks = np.cumsum(1 - correct[order]) / np.arange(1, len(correct) + 1)
    error = 1 - correct.mean()
    # Optimal ranking: all correct answers first.
    optimal = error + (1 - error) * np.log(1 - error) if 0 < error < 1 else 0.0
    area = float(risks.mean())
    return {"aurc": area, "e_aurc": area - float(optimal),
            "accuracy_at_80": float(correct[order[: int(0.8 * len(correct))]].mean()),
            "accuracy_at_90": float(correct[order[: int(0.9 * len(correct))]].mean())}


def compute_curve(records: list[dict], gate: list[float], k: int, selector: str,
                  budgets: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0)) -> dict:
    """Send the lowest-gated fraction b of questions to a selector over pool k; the rest keep greedy."""
    greedy = np.asarray(outcomes(records, k, "greedy"), dtype=float)
    chosen = np.asarray(outcomes(records, k, selector), dtype=float)
    order = np.argsort(np.asarray(gate, dtype=float), kind="stable")
    curve = {}
    for budget in budgets:
        routed = np.zeros(len(records), dtype=bool)
        routed[order[: int(round(budget * len(records)))]] = True
        accuracy = np.where(routed, chosen, greedy).mean()
        curve[f"{budget:.1f}"] = {"accuracy": float(accuracy), "gain": float(accuracy - greedy.mean()),
                                  "generations_per_question": 1 + budget * k}
    gains = [point["gain"] for point in curve.values()]
    curve["mean_gain"] = float(np.mean(gains))
    return curve


def gate_scores(records: list[dict], gate: str) -> list[float]:
    """A question-level gate: the pre-generation score of a head, or the greedy answer's head or token score."""
    if gate.startswith("pre_"):
        return [r["pre_generation"][gate.removeprefix("pre_")] for r in records]
    return [score_of(r["greedy"], gate) for r in records]

