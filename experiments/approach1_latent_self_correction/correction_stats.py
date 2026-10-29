# correction_stats.py
#
# Statistics for the correction experiments on the full test split. Every policy is a way of
# choosing, for each question, between an original answer and a second one, so two policies are
# compared on the same questions. This module gives, for any policy, its accuracy with a Wilson
# interval, how many answers it fixed and broke against the baseline, an exact McNemar p-value
# and a paired bootstrap interval for the accuracy difference, plus the AUROC of a score over
# candidates. It reads the merged records written by merge_correction_shards.py and has no model
# or dataset dependencies, so it runs anywhere.

import math

import numpy as np

RETRY_POLICIES = {
    "baseline_no_retry": "never_selected",
    "retry_all": "retry_all_selected",
    "head_threshold": "head_threshold_selected",
    "head_lowest_fraction": "head_fraction_selected",
    "token_lowest_fraction": "token_fraction_selected",
    "random_matched_fraction": "random_fraction_selected",
    "head_rerank": "head_rerank_selected",
    "head_gated_margin_0.00": "head_gated_margin_0_0_selected",
    "head_gated_margin_0.05": "head_gated_margin_0_05_selected",
    "head_gated_margin_0.10": "head_gated_margin_0_1_selected",
    "head_gated_margin_0.20": "head_gated_margin_0_2_selected",
    "oracle_upper_bound": "oracle_selected",
}
DEEP_POLICIES = {
    "keep_best_of_five": "keep_selected",
    "deep_replace_all": "deep_replace_all",
    "deep_head_rerank": "deep_head_rerank",
    "deep_margin_0.00": "deep_margin_0_0",
    "deep_margin_0.05": "deep_margin_0_05",
    "deep_margin_0.10": "deep_margin_0_1",
    "deep_margin_0.20": "deep_margin_0_2",
    "deep_oracle": "deep_oracle",
}
BEST5_POLICIES = ("greedy", "first_sample", "majority_vote", "token_best", "head_best", "oracle")


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of a proportion."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - spread) / denominator, (centre + spread) / denominator


def mcnemar_p(fixed: int, broken: int) -> float:
    """Exact two-sided p-value of a sign test on the questions where two policies disagree."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_difference(baseline: list, other: list, boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy of a policy against a baseline on the same questions: difference, bootstrap interval, fixed, broken, p."""
    a, b = np.asarray(baseline, dtype=bool), np.asarray(other, dtype=bool)
    fixed, broken = int((~a & b).sum()), int((a & ~b).sum())
    diffs = (b.astype(float) - a.astype(float))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(diffs), size=(boot, len(diffs)))
    means = diffs[draws].mean(axis=1)
    return {"difference": float(diffs.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)),
            "fixed": fixed, "broken": broken, "p_value": mcnemar_p(fixed, broken)}


def auroc(labels, scores) -> float:
    """AUROC from average ranks, so ties count half."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positives = labels.sum()
    negatives = len(labels) - positives
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def retry_outcomes(records: list[dict], key: str) -> tuple[list, list]:
    """(baseline correct, final correct) per question for one retry policy flag."""
    base = [bool(r["baseline_correct"]) for r in records]
    final = [bool(r["retry_correct"]) if r.get(key) else bool(r["baseline_correct"]) for r in records]
    return base, final


def deep_outcomes(records: list[dict], key: str) -> tuple[list, list]:
    """(selected candidate correct, final correct) per question for one deeper-pass policy flag."""
    base = [bool(r["selected_candidate"]["correct"]) for r in records]
    final = [bool(r["deep_answer"]["correct"]) if r.get(key) else bool(r["selected_candidate"]["correct"]) for r in records]
    return base, final


def best5_outcomes(records: list[dict], policy: str) -> list:
    """Whether the answer chosen by one best-of-five policy is correct, per question."""
    if policy == "greedy":
        return [bool(r["greedy"]["correct"]) for r in records]
    return [bool(r["candidates"][r["selected_indices"][policy]]["correct"]) for r in records]


def policy_row(base: list, final: list) -> dict:
    """Accuracy with a Wilson interval and the paired comparison against the baseline."""
    low, high = wilson(sum(final), len(final))
    row = paired_difference(base, final)
    row.update(accuracy=sum(final) / len(final), accuracy_low=low, accuracy_high=high, examples=len(final))
    return row


def format_failures(records: list[dict], token_limit: int = 64) -> dict:
    """Count deeper-pass answers that hit the token limit or repeated the prompt, and how many correct answers they broke."""
    capped = echoed = broken_capped = broken_echoed = 0
    for r in records:
        is_capped = len(r["deep_answer"]["token_probabilities"]) >= token_limit
        is_echo = str(r["deep_answer"]["answer"]).startswith("A candidate answer was")
        broke = r["selected_candidate"]["correct"] and not r["deep_answer"]["correct"]
        capped += int(is_capped)
        echoed += int(is_echo and not is_capped)
        broken_capped += int(is_capped and broke)
        broken_echoed += int(is_echo and not is_capped and broke)
    return {"examples": len(records), "capped": capped, "echoed": echoed, "broken_capped": broken_capped, "broken_echoed": broken_echoed}
