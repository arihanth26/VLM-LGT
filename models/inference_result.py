# inference_result.py
#
# Structured output from standard model inference. Token likelihood is retained
# as a diagnostic measurement, not as a correctness or selection rule.

from dataclasses import dataclass, field
@dataclass
class InferenceResult:
    """A generated answer and confidence statistics for its output tokens."""

    answer: str
    confidence: float
    mean_log_probability: float
    minimum_token_probability: float
    token_probabilities: list[float] = field(default_factory=list)
