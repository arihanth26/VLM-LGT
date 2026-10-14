# build_correction_report.py
#
# Combines the prompted-retry and best-of-five correction experiments into one
# self-contained HTML report with short explanations of every selection method.

import argparse
import html
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "correction_experiments_report.html"
DATASETS = ("chartqa_full", "textvqa", "docvqa", "scienceqa_img")

METHODS = {
    "baseline_no_retry": ("No retry", "Keep the model's original greedy answer."),
    "retry_all": ("Retry all", "Replace every original answer with one visually prompted retry."),
    "head_threshold": ("Head score < 0.5", "Retry only originals whose calibrated unified-head score is below 0.5."),
    "head_lowest_fraction": ("Lowest 20% by head", "Retry the fifth of examples with the lowest unified-head scores."),
    "token_lowest_fraction": ("Lowest 20% by token", "Retry the fifth with the lowest geometric-mean answer-token probability."),
    "random_matched_fraction": ("Random 20%", "Retry a seeded random fifth as a matched-compute control."),
    "head_rerank": ("Head reranking", "Score the original and retry, then keep whichever has the higher unified-head score."),
    "head_gated_margin_0.00": ("Gated reranking, margin 0", "Use the retry only when the original scores below 0.5 and the retry scores higher."),
    "head_gated_margin_0.05": ("Gated reranking, margin 0.05", "Require a low-scored original and at least a 0.05 head-score improvement."),
    "head_gated_margin_0.10": ("Gated reranking, margin 0.10", "Require a low-scored original and at least a 0.10 head-score improvement."),
    "head_gated_margin_0.20": ("Gated reranking, margin 0.20", "Require a low-scored original and at least a 0.20 head-score improvement."),
    "oracle_upper_bound": ("Retry oracle", "Choose the retry exactly when it fixes an original error; this is an unattainable upper bound."),
    "greedy": ("Greedy baseline", "Generate one deterministic answer without sampling."),
    "first_sample": ("First sample", "Use the first of five temperature-sampled candidates without selection."),
    "majority_vote": ("Majority vote", "Choose the most common normalized answer among five samples."),
    "token_best": ("Token-confidence best", "Choose the sampled candidate with the highest answer-token probability."),
    "head_best": ("Unified-head best", "Choose the sampled candidate with the highest learned correctness score."),
    "oracle": ("Candidate oracle", "Choose a correct sampled candidate whenever one exists; this is an unattainable upper bound."),
    "keep_best_of_five": ("Keep best-of-five", "Keep the candidate selected by the unified head without another generation pass."),
    "deep_replace_all": ("Deep pass for all", "Replace every selected candidate with an answer produced after asking the model to think more deeply."),
    "deep_head_rerank": ("Deep-pass reranking", "Score the selected candidate and deeper answer, then keep the one with the higher unified-head score."),
    "deep_margin_0.00": ("Deep margin 0", "Use the deeper answer when its unified-head score is higher."),
    "deep_margin_0.05": ("Deep margin 0.05", "Use the deeper answer only when its score improves by at least 0.05."),
    "deep_margin_0.10": ("Deep margin 0.10", "Use the deeper answer only when its score improves by at least 0.10."),
    "deep_margin_0.20": ("Deep margin 0.20", "Use the deeper answer only when its score improves by at least 0.20."),
    "deep_oracle": ("Deep-pass oracle", "Choose the deeper answer exactly when it fixes the selected candidate; this is an upper bound."),
}


def load_json(path: Path) -> dict:
    """Read one required experiment summary with a clear missing-file error."""
    if not path.exists():
        raise FileNotFoundError(f"missing result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy_table(summary: dict, order: tuple[str, ...], retry_schema: bool) -> str:
    """Render pooled and per-dataset accuracy for one experiment."""
    rows = []
    for key in order:
        if key not in summary["policies"]:
            continue
        label, description = METHODS[key]
        result = summary["policies"][key]
        pooled = result["pooled"]
        cells = [
            f"<td><strong>{html.escape(label)}</strong><br><span class='muted'>{html.escape(description)}</span></td>",
            f"<td>{pooled['accuracy']:.1%}</td>" if not retry_schema else f"<td>{pooled['final_accuracy']:.1%}</td>",
        ]
        if retry_schema:
            cells.extend([
                f"<td>{pooled['flagged']}</td>",
                f"<td class='positive'>+{pooled['fixed_errors']}</td>",
                f"<td class='negative'>-{pooled['broken_answers']}</td>",
            ])
        for dataset in DATASETS:
            metric = result["per_dataset"][dataset]
            value = metric["final_accuracy"] if retry_schema else metric["accuracy"]
            cells.append(f"<td>{value:.1%}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    extra = "<th>Selected</th><th>Fixed</th><th>Broken</th>" if retry_schema else ""
    headings = "".join(f"<th>{html.escape(name)}</th>" for name in DATASETS)
    return f"<table><thead><tr><th>Method</th><th>Pooled</th>{extra}{headings}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def candidate_cases(records: list[dict]) -> str:
    """Show examples where learned best-of-five selection changes greedy correctness."""
    rows = []
    for record in records:
        if "candidates" not in record:
            continue
        chosen = record["candidates"][record["selected_indices"]["head_best"]]
        greedy = record["greedy"]
        if chosen["correct"] == greedy["correct"]:
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['dataset'])}:{record['index']}</td>"
            f"<td>{html.escape(record['question'])}</td>"
            f"<td>{html.escape(str(record['reference_answer']))}</td>"
            f"<td>{html.escape(greedy['answer'])}<br><span class='muted'>{'correct' if greedy['correct'] else 'wrong'}</span></td>"
            f"<td>{html.escape(chosen['answer'])}<br><span class='muted'>head {chosen['head_confidence']:.3f}, {'correct' if chosen['correct'] else 'wrong'}</span></td>"
            "</tr>"
        )
    return "".join(rows)


def build_report(results_dir: Path, output: Path) -> None:
    """Load completed experiments and write the combined correction report."""
    retry = load_json(results_dir / "mixture_test_2b_prompted_retry_limit100.summary.json")
    reranked = load_json(results_dir / "mixture_test_2b_prompted_retry_reranked_limit100.summary.json")
    candidates = load_json(results_dir / "mixture_test_2b_best_of_five_limit100.summary.json")
    deep_path = results_dir / "mixture_test_2b_best_of_five_deep_limit100.summary.json"
    deep = load_json(deep_path) if deep_path.exists() else None
    candidate_records = [
        json.loads(line)
        for line in (results_dir / "mixture_test_2b_best_of_five_limit100.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    retry_order = (
        "baseline_no_retry", "retry_all", "head_threshold", "head_lowest_fraction",
        "token_lowest_fraction", "random_matched_fraction",
    )
    rerank_order = (
        "baseline_no_retry", "retry_all", "head_rerank", "head_gated_margin_0.00",
        "head_gated_margin_0.05", "head_gated_margin_0.10", "head_gated_margin_0.20",
        "oracle_upper_bound",
    )
    candidate_order = ("greedy", "first_sample", "majority_vote", "token_best", "head_best", "oracle")
    deep_order = (
        "keep_best_of_five", "deep_replace_all", "deep_head_rerank", "deep_margin_0.00",
        "deep_margin_0.05", "deep_margin_0.10", "deep_margin_0.20", "deep_oracle",
    )
    cases = candidate_cases(candidate_records)
    deep_section = ""
    if deep is not None:
        deep_section = f"""
<h2>4. Deeper pass after best-of-five selection</h2>
<p>The unified head first selects one of five sampled candidates. Qwen then sees that candidate and is asked to think more deeply, recheck the visual evidence, and provide a corrected final answer. Replacement and score-margin policies decide whether the deeper answer should supersede the selected candidate.</p>
{accuracy_table(deep, deep_order, True)}
"""

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Approach 1 correction experiments</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1500px;margin:40px auto;padding:0 24px;color:#17212b}}h1,h2{{line-height:1.2}}h2{{margin-top:42px}}table{{border-collapse:collapse;width:100%;margin:18px 0 28px}}th,td{{border:1px solid #ccd4dc;padding:9px;vertical-align:top;text-align:left}}th{{background:#edf2f6}}tr:nth-child(even){{background:#fafbfc}}.muted{{color:#64717d;font-size:.9em}}.positive{{color:#147a39}}.negative{{color:#ad2d2d}}.callout{{background:#edf6ff;border-left:5px solid #2878bd;padding:14px 18px;margin:20px 0}}code{{background:#eef1f4;padding:2px 5px;border-radius:4px}}
</style></head><body>
<h1>Approach 1: correction experiments</h1>
<p>All experiments use frozen Qwen3-VL-2B and the exported mixture-trained unified correctness head. Each study contains 100 examples, balanced across ChartQA, TextVQA, DocVQA, and ScienceQA. Accuracy is the dataset-specific metric converted to the binary correctness definition used to train the head.</p>
<div class="callout"><strong>Main result.</strong> Sampling five candidates and selecting with the unified head reaches <strong>{candidates['policies']['head_best']['pooled']['accuracy']:.1%}</strong>, compared with <strong>{candidates['policies']['greedy']['pooled']['accuracy']:.1%}</strong> for greedy generation. The five-candidate oracle is <strong>{candidates['policies']['oracle']['pooled']['accuracy']:.1%}</strong>.</div>

<h2>1. Prompted retry</h2>
<p>The model first answers normally. A second prompt shows the previous answer and asks it to inspect visual evidence again. These policies test when that retry should replace the original.</p>
{accuracy_table(retry, retry_order, True)}
<p class="muted">This first run began at 81% baseline accuracy. Retrying every answer reduced accuracy to 78%; head-based targeting also reached 78%, while the seeded random control reached 82% on this small sample.</p>

<h2>2. Scored retry and reranking</h2>
<p>The same unified head scores both the original answer and its retry. Direct reranking picks the higher score. Gated variants also require the original score to be below 0.5 and optionally require a minimum score improvement.</p>
{accuracy_table(reranked, rerank_order, True)}
<p class="muted">This independently generated run began at 79%. The best practical retry policies reached 80%; the retry oracle reached 83%. Slight baseline variation between runs comes from generation and GPU nondeterminism, so comparisons should be made within each table.</p>

<h2>3. Best-of-five candidate selection</h2>
<p>Five independent answers are sampled at temperature {candidates['temperature']} with top-p {candidates['top_p']}. The learned selector scores every candidate from its answer-conditioned hidden state. Majority vote uses normalized answer text, while token confidence uses the geometric mean of generated-token probabilities.</p>
{accuracy_table(candidates, candidate_order, False)}
<p>Candidate sets contained an average of <strong>{candidates['candidate_diversity']['mean_unique_answers']:.2f}</strong> unique answers; all five samples agreed on <strong>{candidates['candidate_diversity']['unanimous_fraction']:.0%}</strong> of examples. Unified-head selection improved greedy accuracy by three points and captured three of the seven points available above greedy under the oracle.</p>
{deep_section}

<h2>Best-of-five changed outcomes</h2>
<p>Cases below are those where unified-head selection changed whether the answer was correct relative to greedy generation.</p>
<table><thead><tr><th>Example</th><th>Question</th><th>Reference</th><th>Greedy</th><th>Head-selected candidate</th></tr></thead><tbody>{cases or '<tr><td colspan="5">No changed outcomes.</td></tr>'}</tbody></table>

<h2>Interpretation</h2>
<p>A single prompted retry is unstable: it fixes some errors but can overturn correct answers. Scoring the retry limits that damage but provides only a small gain. Best-of-five generation is more promising because it separates candidate creation from candidate selection. The learned head outperforms majority voting and token likelihood, although the 85% oracle shows that candidate selection and candidate diversity both still leave room for improvement.</p>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"[done] wrote {output}")


def main() -> None:
    """Parse result and output locations."""
    parser = argparse.ArgumentParser(description="Build the combined correction experiment report.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
