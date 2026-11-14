# gating.py
#
# Deployable gating, shared by the correction step (gate_ranker_v3.py) and the escalation study
# (analyze_escalation.py): decide per question, from one detector score, whether to spend extra compute.
#
# A gate is a fixed threshold, not a share of the batch. For a target budget b (the share of questions
# to correct) the threshold is the b-quantile of the detector's scores on VALIDATION; a question is
# corrected when its own score is at or below it. Each question's decision therefore depends only on
# itself and on numbers fixed before test, as in deployment, and the share of test questions actually
# corrected is measured, not imposed. Low score = likely wrong = corrected first, for every detector.
#
# The random gate corrects each question independently with probability b, the baseline any detector
# has to beat at the same budget.

import numpy as np

BUDGETS = tuple(round(0.05 * i, 2) for i in range(21))


def threshold(reference: np.ndarray, budget: float) -> float:
    """Score at or below which a question is corrected, so that `budget` of the reference scores are."""
    reference = np.sort(np.asarray(reference, dtype=float))
    count = int(round(budget * len(reference)))
    if count <= 0:
        return -np.inf
    if count >= len(reference):
        return np.inf
    return float(reference[count - 1])


def gate(scores: np.ndarray, limit: float) -> np.ndarray:
    """Which questions a threshold sends to correction."""
    return np.asarray(scores, dtype=float) <= limit


def random_gate(count: int, budget: float, seed: int = 1) -> np.ndarray:
    """Each question corrected independently with probability `budget`."""
    return np.random.default_rng(seed).random(count) < budget


def combine(routed: np.ndarray, corrected: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Final correctness: the corrected outcome where routed, the original elsewhere."""
    return np.where(routed, corrected, original)
