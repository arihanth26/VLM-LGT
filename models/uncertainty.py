# uncertainty.py
#
# Token-likelihood summary helpers. These values are recorded for later
# calibration research but do not decide whether a baseline answer is correct.

import math


def summarize_token_log_probabilities(log_probabilities: list[float]) -> tuple[float, float, float]:
    """Return geometric-mean confidence, mean log probability, and minimum probability."""
    if not log_probabilities:
        return 0.0, float("-inf"), 0.0

    mean_log_probability = sum(log_probabilities) / len(log_probabilities)
    confidence = math.exp(mean_log_probability)
    minimum_probability = math.exp(min(log_probabilities))
    return confidence, mean_log_probability, minimum_probability
