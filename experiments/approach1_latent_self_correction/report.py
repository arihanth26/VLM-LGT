# report.py
#
# Converts baseline JSONL output into a readable Markdown accuracy report.

import argparse
import json
from pathlib import Path


def table_text(value) -> str:
    """Make a value safe for a compact Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def write_markdown_report(input_path: Path, output_path: Path) -> None:
    """Render baseline records and aggregate accuracy as Markdown."""
    with input_path.open("r", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]

    successful = [record for record in records if "prediction" in record]
    correct = sum(bool(record["correct"]) for record in successful)
    accuracy = correct / len(successful) if successful else 0.0
    errors = len(records) - len(successful)
    lines = [
        "# Full-image baseline",
        "",
        f"- Records: {len(records)}",
        f"- Successful: {len(successful)}",
        f"- Correct: {correct}",
        f"- Relaxed accuracy: {accuracy:.2%}",
        f"- Errors: {errors}",
        "",
        "| # | Question | Reference | Prediction | Correct | Confidence |",
        "|---:|---|---|---|:---:|---:|",
    ]
    for record in records:
        if "prediction" not in record:
            lines.append(
                f"| {record['index']} | Error | - | {table_text(record.get('error'))} | - | - |"
            )
            continue
        prediction = record["prediction"]
        lines.append(
            f"| {record['index']} | {table_text(record['question'])} | "
            f"{table_text(record.get('reference_answer'))} | {table_text(prediction['answer'])} | "
            f"{record['correct']} | {prediction['confidence']:.3f} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {output_path}")


def main() -> None:
    """Parse paths and render a baseline report."""
    parser = argparse.ArgumentParser(description="Convert baseline JSONL to Markdown.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    write_markdown_report(args.input, args.output or args.input.with_suffix(".md"))


if __name__ == "__main__":
    main()
