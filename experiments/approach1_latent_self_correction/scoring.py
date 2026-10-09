# scoring.py
#
# Answer scoring for the Q-head datasets (TextVQA, DocVQA, ScienceQA). Each
# dataset has its own official metric, so this file turns every one of them
# into a score in [0, 1] plus a binary "correct" label, which is the target
# the Q-heads are trained on. No torch or datasets imports, so it can be
# tested on its own.
#
#   vqa_accuracy  TextVQA, official VQA soft accuracy over 10 human answers
#   anls          DocVQA, normalized Levenshtein similarity to the best reference
#   mcq_letter    ScienceQA, exact match on the chosen option letter

import re

# Binary label cutoffs. With 10 human answers, a VQA score of 0.5 or more
# needs at least two annotators to agree with the prediction (one match
# scores 0.3, two score 0.6). The ANLS cutoff is stricter than the 0.5 used
# in the official leaderboard so a half-right string is not labeled correct.
VQA_CORRECT_THRESHOLD = 0.5
ANLS_CORRECT_THRESHOLD = 0.9

_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}


def normalize_vqa_answer(text: str) -> str:
    """Lowercase, strip punctuation, drop articles, and map number words to digits."""
    text = str(text).lower().strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^\w\s\.]", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    words = [_NUMBER_WORDS.get(word, word) for word in text.split() if word not in _ARTICLES]
    return " ".join(words)


def vqa_accuracy(prediction: str, answers: list[str]) -> float:
    """Official VQA soft accuracy, averaged over every leave-one-annotator-out subset."""
    normalized = [normalize_vqa_answer(answer) for answer in answers]
    predicted = normalize_vqa_answer(prediction)
    if not normalized:
        return 0.0
    scores = []
    for held_out in range(len(normalized)):
        others = normalized[:held_out] + normalized[held_out + 1 :]
        scores.append(min(1.0, sum(answer == predicted for answer in others) / 3.0))
    return sum(scores) / len(scores)


def levenshtein(first: str, second: str) -> int:
    """Edit distance between two strings using a rolling two-row table."""
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for i, char_a in enumerate(first, start=1):
        current = [i]
        for j, char_b in enumerate(second, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def anls_similarity(prediction: str, answers: list[str]) -> float:
    """Best normalized string similarity (1 minus normalized edit distance) over all references."""
    predicted = " ".join(str(prediction).lower().split())
    best = 0.0
    for answer in answers:
        reference = " ".join(str(answer).lower().split())
        longest = max(len(predicted), len(reference))
        similarity = 1.0 if longest == 0 else 1.0 - levenshtein(predicted, reference) / longest
        best = max(best, similarity)
    return best


def parse_option_letter(prediction: str) -> str | None:
    """Pull the option letter out of a reply such as 'B', '(B)', 'B. Paris', or 'Answer: B'."""
    text = str(prediction).strip()
    match = re.match(r"^\W*([A-Za-z])(?![A-Za-z])", text) or re.search(
        r"answer\W+(?:is\W+)?\(?([A-Ha-h])(?![A-Za-z])", text, re.IGNORECASE
    )
    return match.group(1).upper() if match else None


def score_with_metric(metric: str, answers: list[str], prediction: str) -> tuple[bool, float]:
    """Return (correct, score) for one prediction under a named dataset metric."""
    if metric == "vqa_accuracy":
        score = vqa_accuracy(prediction, answers)
        return score >= VQA_CORRECT_THRESHOLD, score
    if metric == "anls":
        score = anls_similarity(prediction, answers)
        return score >= ANLS_CORRECT_THRESHOLD, score
    if metric == "mcq_letter":
        correct = parse_option_letter(prediction) == str(answers[0]).strip().upper()
        return correct, float(correct)
    raise ValueError(f"unknown metric '{metric}'")
