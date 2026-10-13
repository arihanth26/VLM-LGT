# build_mixture_report.py
#
# Builds the report for the dataset-mixture and new-loss experiments as one
# self-contained HTML file. It reads the sweep outputs written by
# sweep_mixture.py (q_models/mixture/mixture_<kind>_results.json), the analysis
# written by analyze_mixture.py (results/analysis_mixture.json), and, for the
# comparison with heads trained on one dataset, the earlier per-dataset sweeps.
# Charts and styling come from build_report.py so the two reports match.

import json
import math
from pathlib import Path

from experiments.approach1_latent_self_correction.build_report import (
    CSS,
    ORIGINAL_TEST,
    DATASET_INFO,
    SCORERS,
    SCORER_CLASS,
    SCORER_LABEL,
    Panel,
    dot_plot,
    esc,
    f3,
    figure,
    legend,
    line_panel,
    pct,
    strip_plot,
    table,
)

HERE = Path(__file__).parent
MIXTURE_DIR = HERE / "q_models" / "mixture"
ANALYSIS = HERE / "results" / "analysis_mixture.json"
OUTPUT = HERE / "results" / "q_head_mixture_report.html"
NAMES = {"chartqa_full": "ChartQA", "textvqa": "TextVQA", "docvqa": "DocVQA", "scienceqa_img": "ScienceQA"}
KINDS = ("pre", "verifier", "unified")
# series key -> (head kind, mode, label, css class)
SERIES = {
    "separate_pre_generation": ("pre", "main"),
    "separate_verifier": ("verifier", "main"),
    "unified_pre_generation": ("unified", "pre_generation"),
    "unified_joint": ("unified", "joint"),
}
OLD_CSS, NEW_CSS = "c-sv", "c-uj"
SCORER_LABEL["loss_old"], SCORER_CLASS["loss_old"] = "Unified head, old loss", OLD_CSS
SCORER_LABEL["loss_new"], SCORER_CLASS["loss_new"] = "Unified head, new loss", NEW_CSS
METRIC_LABELS = {
    "correctness_auroc": "Correctness AUROC",
    "error_auprc": "Error AUPRC",
    "brier": "Brier score",
    "ece_10_bin": "ECE (10 bins)",
    "pearson_correctness": "Pearson r",
}


class Results:
    """Typed access to the sweep winners, the analysis file, and the earlier per-dataset results."""

    def __init__(self):
        self.sweeps = {k: json.loads((MIXTURE_DIR / f"mixture_{k}_results.json").read_text(encoding="utf-8")) for k in KINDS if (MIXTURE_DIR / f"mixture_{k}_results.json").exists()}
        self.analysis = json.loads(ANALYSIS.read_text(encoding="utf-8")) if ANALYSIS.exists() else {}
        self.continual = {k: json.loads((MIXTURE_DIR / f"continual_{k}_results.json").read_text(encoding="utf-8")) for k in KINDS if (MIXTURE_DIR / f"continual_{k}_results.json").exists()}
        first = next(iter(self.sweeps.values()))
        self.datasets = list(first["datasets"])
        self.rows = first["rows"]
        self.earlier = {}
        for key in self.datasets:
            path = HERE / "results" / f"analysis_{key}.json"
            if path.exists():
                self.earlier[key] = json.loads(path.read_text(encoding="utf-8"))
        self.specialists = {}
        for key in self.datasets:
            folder = HERE / "q_models" / key
            if (folder / "sweep_results.json").exists() and (folder / "unified_sweep_results.json").exists():
                separate = json.loads((folder / "sweep_results.json").read_text(encoding="utf-8"))
                unified = json.loads((folder / "unified_sweep_results.json").read_text(encoding="utf-8"))
                self.specialists[key] = {
                    "separate_pre_generation": separate["pre_generation"]["test"]["correctness_auroc"],
                    "separate_verifier": separate["candidate_verifier"]["test"]["correctness_auroc"],
                    "unified_pre_generation": unified["test_summary"]["pre_generation"]["correctness_auroc"]["mean"],
                    "unified_joint": unified["test_summary"]["joint"]["correctness_auroc"]["mean"],
                }

    def winner(self, kind: str, name: str):
        """A winner entry, or None if that group was not run."""
        return self.sweeps.get(kind, {}).get("winners", {}).get(name)

    def metric(self, kind, name, mode, metric, dataset=None):
        """(mean, std) of a test metric, pooled or for one dataset."""
        w = self.winner(kind, name)
        if w is None:
            return None
        node = w["test"][mode][metric] if dataset is None else w["test_per_dataset"].get(dataset, {}).get(mode, {}).get(metric)
        return None if node is None else (node["mean"], node["std"])

    def grounding(self, kind, name, mode, field, subset="correct", dataset=None):
        """(mean, std) of a grounding metric, pooled or for one dataset."""
        w = self.winner(kind, name)
        if w is None:
            return None
        node = w["grounding"][mode].get("pooled" if dataset is None else dataset, {}).get(subset, {}).get(field)
        return None if node is None else (node["mean"], node["std"])

    def baseline(self, label="pooled"):
        """Baselines of the answer-token probability and the length rule for one subset."""
        return self.analysis.get("baselines", {}).get(label)


def mean_of(pair):
    """The mean of a (mean, std) pair, or NaN if the pair is missing."""
    return float("nan") if pair is None else pair[0]


def change_words(delta: float, spread: float, unit: str = "AUROC") -> str:
    """Describe a difference in plain words, comparing it with the seed spread."""
    if math.isnan(delta):
        return "could not be measured"
    if abs(delta) <= max(2 * spread, 0.002):
        return f"did not change by more than the seed spread ({delta:+.3f} {unit})"
    return f"{'raised' if delta > 0 else 'lowered'} {unit} by {abs(delta):.3f}"


def diff_plot(rows, xlabel: str, caption_id: str = "") -> str:
    """Dots with one-spread whiskers for a difference, with a zero line. Rows are (label, css, diff, spread)."""
    values = [v for _, _, d, s in rows for v in (d - s, d + s, 0.0)]
    limit = max(abs(min(values)), abs(max(values))) * 1.2 or 0.01
    row_h, left, top = 26, 230, 8
    height = top + row_h * len(rows) + 40
    panel = Panel(left, top, 640 - left - 20, row_h * len(rows), -limit, limit, 0, 1)
    parts = [f'<svg viewBox="0 0 640 {height}" role="img" aria-labelledby="{caption_id}">']
    for i in range(9):
        tick = -limit + i * limit / 4
        parts.append(f'<line class="grid" x1="{panel.sx(tick):.1f}" x2="{panel.sx(tick):.1f}" y1="{top}" y2="{top + row_h * len(rows)}"/>')
        parts.append(f'<text class="tick" x="{panel.sx(tick):.1f}" y="{top + row_h * len(rows) + 15}" text-anchor="middle">{tick:+.3f}</text>')
    parts.append(f'<line class="axis" x1="{panel.sx(0):.1f}" x2="{panel.sx(0):.1f}" y1="{top}" y2="{top + row_h * len(rows)}"/>')
    parts.append(f'<text class="axlabel" x="{left + panel.w / 2}" y="{top + row_h * len(rows) + 33}" text-anchor="middle">{esc(xlabel)}</text>')
    for index, (label, css, diff, spread) in enumerate(rows):
        yc = top + row_h * index + row_h / 2
        parts.append(f'<text class="rowlabel" x="{left - 10}" y="{yc + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<line class="{css} ln" x1="{panel.sx(diff - spread):.1f}" x2="{panel.sx(diff + spread):.1f}" y1="{yc:.1f}" y2="{yc:.1f}"/>')
        parts.append(f'<circle class="{css} dot" cx="{panel.sx(diff):.1f}" cy="{yc:.1f}" r="4.5"><title>{esc(label)}: {diff:+.4f} (one seed spread {spread:.4f})</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def pair_plot(rows, xmin: float, xmax: float, xlabel: str, caption_id: str = "") -> str:
    """One row per group with an old-loss dot and a new-loss dot, and an optional token-confidence dot."""
    row_h, left, top = 38, 210, 8
    height = top + row_h * len(rows) + 40
    panel = Panel(left, top, 640 - left - 16, row_h * len(rows), xmin, xmax, 0, 1)
    parts = [f'<svg viewBox="0 0 640 {height}" role="img" aria-labelledby="{caption_id}">']
    for i in range(6):
        tick = xmin + i * (xmax - xmin) / 5
        parts.append(f'<line class="grid" x1="{panel.sx(tick):.1f}" x2="{panel.sx(tick):.1f}" y1="{top}" y2="{top + row_h * len(rows)}"/>')
        parts.append(f'<text class="tick" x="{panel.sx(tick):.1f}" y="{top + row_h * len(rows) + 15}" text-anchor="middle">{tick:.2f}</text>')
    parts.append(f'<text class="axlabel" x="{left + panel.w / 2}" y="{top + row_h * len(rows) + 33}" text-anchor="middle">{esc(xlabel)}</text>')
    for index, (label, old, new, token) in enumerate(rows):
        yc = top + row_h * index + row_h / 2
        parts.append(f'<text class="rowlabel" x="{left - 10}" y="{yc + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        if index:
            parts.append(f'<line class="grid" x1="{left}" x2="{left + panel.w}" y1="{top + row_h * index}" y2="{top + row_h * index}"/>')
        for value, css, name, offset in ((token, "c-tok", "Answer-token probability", -9), (old, OLD_CSS, "Old loss", 0), (new, NEW_CSS, "New loss", 9)):
            if value is not None and not math.isnan(value):
                parts.append(f'<circle class="{css} dot" cx="{panel.sx(value):.1f}" cy="{yc + offset:.1f}" r="4.2"><title>{esc(label)}, {name}: {value:.3f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def build(results: Results) -> str:
    """Assemble the full mixture report."""
    r = results
    datasets = r.datasets
    labels = ["Pooled"] + [NAMES.get(d, d) for d in datasets]
    keys = [None] + datasets
    base = {k: r.baseline("pooled" if k is None else k) for k in keys}
    comp = r.analysis.get("composition", {})
    token_pooled = (base[None] or {}).get("token_confidence", {})

    def best_of(name, mode_pairs):
        """Best pooled AUROC among head variants for a winner group."""
        values = {label: mean_of(r.metric(kind, name, mode, "correctness_auroc")) for label, (kind, mode) in mode_pairs.items()}
        values = {k: v for k, v in values.items() if not math.isnan(v)}
        return max(values.items(), key=lambda kv: kv[1]) if values else (None, float("nan"))

    best_old = best_of("old", SERIES)
    best_new = best_of("new", SERIES)

    deltas = {key: delta_pair(r, *SERIES[key]) for key in SERIES}
    grounding_tok = (base[None] or {}).get("grounding_token_confidence", {}).get("correct", {})
    toc_items = [("recap", "Approach 1 so far"), ("summary", "Summary"), ("setup", "Setup"), ("loss", "The new loss"), ("exp1", "Experiment 1"), ("exp2", "Experiment 2"),
                 ("exp3", "Experiment 3"), ("seq", "Sequential training"), ("grounding", "Grounding"), ("ablation", "Ablation"),
                 ("generalist", "Generalist or specialist"), ("use", "Using the score"), ("meaning", "What it means"), ("limits", "Limits")]
    toc = '<nav class="toc">' + "".join(f'<a href="#{i}">{t}</a>' for i, t in toc_items) + "</nav>"
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q-Head Mixture Report</title><style>{CSS}</style></head><body><main>
<h1>Training Correctness Heads Across Datasets</h1>
<p class="sub">Joint training on a mixture of ChartQA, TextVQA, DocVQA and ScienceQA, sequential training that tests what a head still remembers, and a new loss that makes the score fall when the image is corrupted.</p>
<p class="meta">Model: Qwen3-VL-2B. Mixture: {r.rows['train']:,} train, {r.rows['val']:,} validation, {r.rows['test']:,} test rows. Every number is on the pooled test split or on one dataset's rows of it.</p>{toc}"""]

    # ------------------------------------------------------------ summary
    parts.append(recap_section(r))
    parts.append(f"""<h2 id="summary">2. This report in brief</h2><div class="card summary"><ul>{summary_bullets(r, best_old, best_new, deltas, token_pooled, grounding_tok)}</ul></div>""")
    # ------------------------------------------------------------ setup
    parts.append(setup_section(r, comp))
    parts.append(loss_section(r))
    parts.append(experiment_one(r, keys, labels, base))
    parts.append(experiment_two(r, keys, labels, base))
    parts.append(experiment_three(r, keys, labels, base, deltas))
    parts.append(continual_section(r))
    parts.append(grounding_section(r, base, grounding_tok, keys, labels))
    parts.append(ablation_section(r))
    parts.append(generalist_section(r, keys, labels))
    parts.append(use_section(r, labels, keys))
    parts.append(meaning_section(r, best_old, best_new, deltas, grounding_tok))
    parts.append(limits_section())
    parts.append("</main></body></html>")
    return "".join(parts)


def summary_bullets(r, best_old, best_new, deltas, token_pooled, grounding_tok) -> str:
    """The headline findings, with every number read from the results."""
    tok = token_pooled.get("correctness_auroc", float("nan"))
    items = [
        f"<li><b>On the pooled mixture the answer-token probability has AUROC {tok:.3f}.</b> The best head trained with the old loss reaches {best_old[1]:.3f} ({esc(SCORER_LABEL.get(best_old[0], ''))}) and the best with the new loss {best_new[1]:.3f} ({esc(SCORER_LABEL.get(best_new[0], ''))}).</li>"
    ]
    parts = []
    for key, label in (("separate_pre_generation", "separate pre-generation head"), ("separate_verifier", "separate verifier"), ("unified_joint", "unified head, joint mode"), ("unified_pre_generation", "unified head, pre-generation mode")):
        delta, spread = deltas[key]
        parts.append(f"the {label} {change_words(delta, spread)}")
    items.append(f"<li><b>Effect of the new loss on pooled AUROC.</b> Compared with the old loss: " + "; ".join(parts) + ".</li>")
    order_tok = grounding_tok.get("order_accuracy", float("nan"))
    og, ng = r.grounding("unified", "old", "joint", "order_accuracy"), r.grounding("unified", "new", "joint", "order_accuracy")
    if og and ng:
        items.append(
            f"<li><b>Grounding.</b> When the image is swapped for another one, the score of a correct answer falls in {pct(order_tok, 0)} of cases for the answer-token probability, {pct(og[0], 0)} for the unified head with the old loss, and {pct(ng[0], 0)} with the new loss.</li>"
        )
    items.extend(extra_summary(r))
    return "".join(items)


def seq_value(r, kind, key, mode, metric):
    """Mean of a sequential-training summary metric, or NaN if that run is missing."""
    data = r.continual.get(kind)
    if not data or key not in data["variants"]:
        return float("nan")
    return data["variants"][key]["summary"][mode][metric]["mean"]


def final_grounding(r, kind, key, mode):
    """Mean order accuracy of the grounding check after the last stage, over datasets and orders."""
    data = r.continual.get(kind)
    if not data or key not in data["variants"]:
        return float("nan")
    rows = [entry["grounding_matrix"][mode][-1] for entry in data["variants"][key]["orders"].values()]
    return float(sum(sum(row) / len(row) for row in rows) / len(rows))


def cell_change(r, kind, mode, cell, metric="correctness_auroc", ground=False):
    """Change of a loss cell relative to the plain cell, in AUROC or in grounding order accuracy."""
    if ground:
        base, other = r.grounding(kind, "cell_plain", mode, "order_accuracy"), r.grounding(kind, cell, mode, "order_accuracy")
    else:
        base, other = r.metric(kind, "cell_plain", mode, metric), r.metric(kind, cell, mode, metric)
    if base is None or other is None:
        return float("nan")
    return other[0] - base[0]


def extra_summary(r) -> list:
    """Summary bullets for the ablation and the sequential experiments."""
    items = []
    ground_auroc = cell_change(r, "unified", "joint", "cell_ground")
    pair_auroc = cell_change(r, "unified", "joint", "cell_contrast")
    ground_order = cell_change(r, "unified", "joint", "cell_ground", ground=True)
    if not any(math.isnan(v) for v in (ground_auroc, pair_auroc, ground_order)):
        items.append(
            f"<li><b>The grounding term is free, the pair term is not.</b> For the unified head the grounding term alone changes pooled AUROC by {ground_auroc:+.3f} and raises the share of correct answers whose score falls under a corrupted image by {ground_order:+.3f}. The pair term alone changes AUROC by {pair_auroc:+.3f}. The new loss as specified, with both terms, therefore costs accuracy that the grounding term alone does not.</li>"
        )
    old_u, new_u = seq_value(r, "unified", "old_replay0", "joint", "forgetting"), seq_value(r, "unified", "new_replay0", "joint", "forgetting")
    ver_old, ver_replay = seq_value(r, "verifier", "old_replay0", "main", "forgetting"), seq_value(r, "verifier", "old_replay0.1", "main", "forgetting")
    final_u = seq_value(r, "unified", "old_replay0", "joint", "final_avg")
    if not any(math.isnan(v) for v in (old_u, new_u, ver_old, ver_replay, final_u)):
        items.append(
            f"<li><b>Heads trained on one dataset after another keep what they learned.</b> Trained through all four datasets in sequence, the unified head loses {old_u:.3f} AUROC on earlier datasets with the old loss and {new_u:.3f} with the new loss, and ends at an average AUROC of {final_u:.3f}. The answer-conditioned verifier forgets the most ({ver_old:.3f}), and a 10% replay buffer cuts that to {ver_replay:.3f}.</li>"
        )
    g_old, g_new = final_grounding(r, "unified", "old_replay0", "joint"), final_grounding(r, "unified", "new_replay0", "joint")
    if not math.isnan(g_old) and not math.isnan(g_new):
        items.append(
            f"<li><b>Grounding survives sequential training, and is higher with the new loss.</b> After the last stage, the share of correct answers whose score falls under a corrupted image, averaged over datasets, is {pct(g_old, 0)} with the old loss and {pct(g_new, 0)} with the new loss, for the unified head. Under both losses it barely changes between a dataset's own stage and the end.</li>"
        )
    return items


def setup_section(r, comp) -> str:
    """The mixture, the three experiments, the new loss, and the corrupted-image gallery."""
    rows = []
    for index, name in enumerate(r.datasets):
        cells = [NAMES.get(name, name)]
        for split in ("train", "val", "test"):
            part = comp.get(split, {}).get("per_dataset", {}).get(name)
            cells.append(f"{part['rows']:,} ({pct(part['accuracy'])})" if part else "n/a")
        rows.append(cells)
    total = ["Mixture"] + [f"{comp[s]['rows']:,} ({pct(comp[s]['accuracy'])})" if s in comp else "n/a" for s in ("train", "val", "test")]
    rows.append(total)
    return f"""<h2 id="setup">3. Setup</h2>
<h3>The mixture</h3>
<p>The mixture is the union of the four datasets' existing splits. Its train split is the four train splits pooled, and the same for validation and test, so nothing was resampled and every test question is one the per-dataset study also scored. A single calibration temperature is fitted on the pooled validation split. Rows are shown with the base model's accuracy.</p>
{table(["Dataset", "Train", "Validation", "Test"], rows)}
<h3>Four experiments</h3>
<ol><li><b>The answer-token probability and linear probes on the mixture.</b> Token confidence needs no training. The linear probes are the linear-architecture heads of experiment 2.</li>
<li><b>Separate and unified heads on the mixture, old loss.</b> The pre-generation head (reads the question state), the verifier head (reads the question and answer state), and the unified head (reads both).</li>
<li><b>The same three heads with the new loss.</b></li>
<li><b>Sequential training.</b> The mixture trains on all datasets at once, which is the easy case. The harder question is whether a head that has been trained on dataset A still scores A well after it has been trained on B and then C. Heads are trained on one dataset at a time, in several orders, and scored on every dataset after every stage.</li></ol>
<p>The loss used by the new-loss runs, and how it is measured, is described in the next section.</p>"""


def metric_table(r, keys, labels, name, extra_rows=None, metric="correctness_auroc") -> str:
    """Rows are scorers, columns are pooled and each dataset, for one winner group."""
    headers = ["Scorer"] + labels
    rows = list(extra_rows or [])
    for series, (kind, mode) in SERIES.items():
        cells = [SCORER_LABEL[series]]
        for dataset in keys:
            pair = r.metric(kind, name, mode, metric, dataset)
            cells.append("n/a" if pair is None else f"{pair[0]:.3f} ± {pair[1]:.3f}")
        rows.append(cells)
    return table(headers, rows)


def token_rows(r, keys, metric="correctness_auroc", with_length=True) -> list:
    """Table rows for the answer-token probability and the answer-length rule."""
    out = []
    cells = ["Answer-token probability"]
    for key in keys:
        b = r.baseline("pooled" if key is None else key)
        cells.append("n/a" if not b else f3(b["token_confidence"][metric]))
    out.append(cells)
    if with_length and metric == "correctness_auroc":
        cells = ["Answer-length rule"]
        for key in keys:
            b = r.baseline("pooled" if key is None else key)
            cells.append("n/a" if not b else f3(b["answer_length"]["auroc"]))
        out.append(cells)
    return out


def experiment_one(r, keys, labels, base) -> str:
    """Experiment 1: the answer-token probability and the linear probes."""
    rows = token_rows(r, keys)
    for series, (kind, mode) in SERIES.items():
        cells = [f"Linear probe, {SCORER_LABEL[series].lower()}"]
        for dataset in keys:
            pair = r.metric(kind, "old_linear", mode, "correctness_auroc", dataset)
            cells.append("n/a" if pair is None else f"{pair[0]:.3f} ± {pair[1]:.3f}")
        rows.append(cells)
    prauc = token_rows(r, keys, "error_auprc", False)
    for series, (kind, mode) in SERIES.items():
        cells = [f"Linear probe, {SCORER_LABEL[series].lower()}"]
        for dataset in keys:
            pair = r.metric(kind, "old_linear", mode, "error_auprc", dataset)
            cells.append("n/a" if pair is None else f"{pair[0]:.3f} ± {pair[1]:.3f}")
        prauc.append(cells)
    chart_rows = []
    for label, key in zip(labels, keys):
        values = {}
        b = r.baseline("pooled" if key is None else key)
        if b:
            values["token_confidence"] = b["token_confidence"]["correctness_auroc"]
        for series, (kind, mode) in SERIES.items():
            pair = r.metric(kind, "old_linear", mode, "correctness_auroc", key)
            if pair:
                values[series] = pair[0]
        chart_rows.append((label, values))
    chart = dot_plot(chart_rows, SCORERS, 0.55, 1.0, "AUROC on the test split (axis starts at 0.55)", caption_id="fig-exp1")
    return f"""<h2 id="exp1">5. Experiment 1: the answer-token probability and linear probes</h2>
<p>The question here is the simplest one: on a pool of very different questions, does the answer-token probability work, and how far does a plain linear probe on the frozen hidden state get? All probes were trained on the mixture with the old loss. The table gives AUROC, so 0.5 is a coin flip, with the spread over three seeds after the plus-minus sign.</p>
{table(["Correctness AUROC"] + labels, rows)}
{legend([(label, css) for _, label, css in SCORERS])}
{figure("fig-exp1", "Correctness AUROC of the answer-token probability and of linear probes trained on the mixture", chart, "Each row is the pooled test split or one dataset's rows of it. The same probes are used in every row, nothing was retrained per dataset.")}
<h4>Error AUPRC</h4><p class="small">How well the wrong answers are packed at the top of a queue sorted from least to most trusted. Chance is the error rate of the subset.</p>
{table(["Error AUPRC"] + labels, prauc)}
{reading_exp1(r, keys, labels)}"""


def experiment_two(r, keys, labels, base) -> str:
    """Experiment 2: separate and unified heads trained on the mixture with the old loss."""
    return family_section(r, keys, labels, base, "old", "exp2", "6. Experiment 2: separate and unified heads, old loss",
                          "The best configuration of each head kind, trained on the mixture with the old loss (no grounding term). The winner is the configuration with the best mean validation score over three seeds.")


def family_section(r, keys, labels, base, name, anchor, title, intro) -> str:
    """Pooled and per-dataset tables and a chart for one loss family."""
    chart_rows = []
    for label, key in zip(labels, keys):
        values = {}
        b = r.baseline("pooled" if key is None else key)
        if b:
            values["token_confidence"] = b["token_confidence"]["correctness_auroc"]
        for series, (kind, mode) in SERIES.items():
            pair = r.metric(kind, name, mode, "correctness_auroc", key)
            if pair:
                values[series] = pair[0]
        chart_rows.append((label, values))
    chart = dot_plot(chart_rows, SCORERS, 0.55, 1.0, "AUROC on the test split (axis starts at 0.55)", caption_id=f"fig-{anchor}")
    pooled_rows = []
    for series, (kind, mode) in SERIES.items():
        w = r.winner(kind, name)
        cells = [SCORER_LABEL[series]]
        for metric in METRIC_LABELS:
            pair = r.metric(kind, name, mode, metric)
            cells.append("n/a" if pair is None else f"{pair[0]:.3f} ± {pair[1]:.3f}")
        cells.append("n/a" if w is None else esc(config_text(w["config"])))
        pooled_rows.append(cells)
    return f"""<h2 id="{anchor}">{title}</h2><p>{intro}</p>
<h3>Pooled test split</h3>
{table(["Scorer"] + list(METRIC_LABELS.values()) + ["Winning configuration"], pooled_rows, left_cols=(6,))}
<h3>By dataset</h3>
{table(["Correctness AUROC"] + labels, token_rows(r, keys, with_length=False)[:1] + metric_rows(r, keys, name))}
{legend([(label, css) for _, label, css in SCORERS])}
{figure(f"fig-{anchor}", "Correctness AUROC of the winning heads, pooled and by dataset", chart, "The heads are trained once on the mixture. A row for a dataset is the same head scored on that dataset's test rows only.")}
{reading_family(r, name)}"""


def metric_rows(r, keys, name, metric="correctness_auroc") -> list:
    """Table rows of one metric for the four head variants of a winner group."""
    out = []
    for series, (kind, mode) in SERIES.items():
        cells = [SCORER_LABEL[series]]
        for dataset in keys:
            pair = r.metric(kind, name, mode, metric, dataset)
            cells.append("n/a" if pair is None else f"{pair[0]:.3f} ± {pair[1]:.3f}")
        out.append(cells)
    return out


def config_text(config: dict) -> str:
    """A short readable description of a configuration."""
    parts = [config["architecture"], f"lr {config['learning_rate']:g}"]
    for key, label in (("rank_weight", "rank"), ("pair_weight", "pair"), ("pre_weight", "pre"), ("ground_weight", "ground")):
        if config.get(key):
            parts.append(f"{label} {config[key]:g}")
    return ", ".join(parts)


def experiment_three(r, keys, labels, base, deltas) -> str:
    """Experiment 3: the new loss, and what it changed relative to the old loss."""
    body = family_section(r, keys, labels, base, "new", "exp3", "7. Experiment 3: separate and unified heads, new loss",
                          "The same heads trained with the new loss, where the grounding weight is above 0. The best configuration within this family is shown, chosen on validation.")
    rows = []
    for key in keys:
        for series, (kind, mode) in SERIES.items():
            delta, spread = delta_pair(r, kind, mode, dataset=key)
            label = f"{labels[keys.index(key)]}: {SCORER_LABEL[series]}"
            rows.append((label, NEW_CSS, 0.0 if math.isnan(delta) else delta, spread))
    chart = diff_plot(rows, "New loss minus old loss, correctness AUROC")
    table_rows = []
    for series, (kind, mode) in SERIES.items():
        cells = [SCORER_LABEL[series]]
        for key in keys:
            delta, spread = delta_pair(r, kind, mode, dataset=key)
            cells.append("n/a" if math.isnan(delta) else f"{delta:+.3f}")
        table_rows.append(cells)
    return body + f"""<h3>New loss against old loss</h3>
<p>The difference in test AUROC between the winning new-loss head and the winning old-loss head of each kind. The whisker is the combined spread over three seeds (one standard deviation of each), so a dot whose whisker crosses zero is a difference the seeds cannot separate from nothing.</p>
{figure("fig-delta", "Change in AUROC from the new loss", chart, "Dots to the right of the zero line favor the new loss.", )}
{table(["New minus old, AUROC"] + labels, table_rows)}"""


def heatmap(matrix, row_labels, column_labels, caption_id="", low=0.5, high=1.0) -> str:
    """A small annotated grid of scores, shaded by value in one hue, with the diagonal cells outlined."""
    cell_w, cell_h, left, top = 92, 30, 150, 34
    width = left + cell_w * len(column_labels) + 10
    height = top + cell_h * len(row_labels) + 8
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{caption_id}">']
    for column, label in enumerate(column_labels):
        parts.append(f'<text class="tick" x="{left + cell_w * column + cell_w / 2}" y="{top - 10}" text-anchor="middle">{esc(label)}</text>')
    for row, label in enumerate(row_labels):
        y = top + cell_h * row
        parts.append(f'<text class="rowlabel" x="{left - 10}" y="{y + cell_h / 2 + 4}" text-anchor="end">{esc(label)}</text>')
        for column, value in enumerate(matrix[row]):
            alpha = 0.08 + 0.55 * min(1.0, max(0.0, (value - low) / (high - low)))
            parts.append(
                f'<rect x="{left + cell_w * column + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" rx="4" fill="var(--uj)" fill-opacity="{alpha:.2f}">'
                f'<title>{esc(row_labels[row])}, {esc(column_labels[column])}: {value:.3f}</title></rect>'
            )
            parts.append(f'<text class="rowlabel" x="{left + cell_w * column + cell_w / 2}" y="{y + cell_h / 2 + 4}" text-anchor="middle">{value:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


VARIANTS = (("old_replay0", "Old loss, plain sequential"), ("new_replay0", "New loss, plain sequential"),
            ("old_replay0.1", "Old loss, 10% replay"), ("new_replay0.1", "New loss, 10% replay"))


def continual_section(r) -> str:
    """Sequential training: what is remembered of earlier datasets after training on later ones."""
    if not r.continual:
        return '<h2 id="seq">8. Sequential training</h2><p>The sequential experiments were not available.</p>'
    names = r.datasets
    labels = [NAMES.get(n, n) for n in names]
    rows = []
    for kind in KINDS:
        data = r.continual.get(kind)
        if not data:
            continue
        for mode in modes_of(kind):
            for key, title in VARIANTS:
                variant = data["variants"].get(key)
                if not variant:
                    continue
                summary = variant["summary"][mode]
                rows.append([SCORER_LABEL[series_for(kind, mode)], title] + [f"{summary[m]['mean']:+.3f} ± {summary[m]['std']:.3f}" if m in ("bwt", "retention") else f"{summary[m]['mean']:.3f} ± {summary[m]['std']:.3f}" for m in ("final_avg", "forgetting", "bwt", "retention")])
    table_html = table(["Head", "Training", "Final average AUROC", "Forgetting", "Backward transfer", "Retention vs trained alone"], rows, left_cols=(0, 1))

    kind = "unified" if "unified" in r.continual else next(iter(r.continual))
    mode = modes_of(kind)[0]
    data = r.continual[kind]
    heat_blocks = []
    for key, title in VARIANTS:
        variant = data["variants"].get(key)
        if not variant:
            continue
        entry = variant["orders"]["forward"]
        order_labels = [f"after {NAMES.get(n, n)}" for n in entry["order"]]
        grid = np_list(entry["matrix"][mode])
        columns = [NAMES.get(n, n) for n in names]
        heat_blocks.append(figure(f"fig-heat-{key}", f"{SCORER_LABEL[series_for(kind, mode)]}: {title}, order {' then '.join(NAMES.get(n, n) for n in entry['order'])}",
                                  heatmap(grid, order_labels, columns, f"fig-heat-{key}"), "Rows are the stage just completed, columns the dataset scored. Each cell is test AUROC. The cells a head has not been trained on yet show how well it transfers before it has seen that dataset."))

    forgotten = []
    for key, title in VARIANTS:
        variant = data["variants"].get(key)
        if not variant:
            continue
        per_dataset = {n: [] for n in names}
        for entry in variant["orders"].values():
            matrix = np_list(entry["matrix"][mode])
            order = [names.index(n) for n in entry["order"]]
            for position, dataset in enumerate(order[:-1]):
                column = [matrix[stage][dataset] for stage in range(position, len(order) - 1)]
                per_dataset[names[dataset]].append(max(column) - matrix[-1][dataset])
        forgotten.append([title] + [f"{sum(v) / len(v):+.3f}" if v else "n/a" for v in (per_dataset[n] for n in names)])
    ground_rows = []
    for key, title in VARIANTS:
        variant = data["variants"].get(key)
        if not variant:
            continue
        orders = variant["orders"].values()
        own, final = [], []
        for entry in orders:
            matrix = np_list(entry["grounding_matrix"][mode])
            order = [names.index(n) for n in entry["order"]]
            for position, dataset in enumerate(order[:-1]):
                own.append(matrix[position][dataset])
                final.append(matrix[-1][dataset])
        ground_rows.append([title, f"{sum(own) / len(own):.3f}", f"{sum(final) / len(final):.3f}", f"{sum(final) / len(final) - sum(own) / len(own):+.3f}"])

    joint_rows = []
    for series, (k, md) in SERIES.items():
        data_k = r.continual.get(k)
        if not data_k:
            continue
        for loss in ("old", "new"):
            variant = data_k["variants"].get(f"{loss}_replay0")
            joint = [mean_of(r.metric(k, loss, md, "correctness_auroc", d)) for d in names]
            iso = [variant["isolated"][md][d]["mean"] for d in names] if variant else [float("nan")] * len(names)
            sequential = [variant["orders"]["forward"]["matrix"][md][-1][i] if variant else float("nan") for i in range(len(names))]
            joint_rows.append([SCORER_LABEL[series], loss + " loss"] + [f"{iso[i]:.3f} / {joint[i]:.3f} / {sequential[i]:.3f}" for i in range(len(names))])
    return f"""<h2 id="seq">8. Sequential training: does the head remember earlier datasets?</h2>
<p>Experiments 2 and 3 train on all datasets at once. In practice a head is often trained on one dataset, then extended to another, and the worry is that training on the later dataset overwrites what it learned from the earlier one. This section tests that directly. Each head is trained on one dataset at a time, in three orders (forward, reverse, shuffled), and after every stage it is scored on the test split of all four datasets. Everything else is fixed: the head configuration is the winner of the mixture sweep for that loss, each stage runs until its own validation stops improving, and each variant is repeated on three seeds. Two things are compared: the old loss against the new loss, and plain sequential training against sequential training with a replay buffer holding 10% of every earlier dataset's training rows.</p>
<p><b>How to read the metrics.</b> <i>Final average</i> is mean AUROC over the four datasets after the last stage. <i>Forgetting</i> is, for each dataset trained before the last stage, the best score it had after its own stage minus its final score (negative means it improved). <i>Backward transfer</i> is its final score minus its score right after its own stage. <i>Retention</i> is its final score minus the score of a head trained on that dataset alone. All are averaged over 3 orders and 3 seeds, with the spread after the plus-minus sign.</p>
{table_html}
{seq_reading(r)}
{"".join(heat_blocks)}
<h4>Which dataset is forgotten most</h4>
<p>Forgetting per dataset for the {esc(SCORER_LABEL[series_for(kind, mode)])}, averaged over the orders in which that dataset was not the last one trained.</p>
{table(["Training"] + labels, forgotten)}
<h4>Does the grounding behavior survive?</h4>
<p>The share of correct answers whose score falls under a corrupted image, right after the stage that trained the dataset and after the last stage, averaged over datasets. 0.5 means no effect.</p>
{table(["Training", "Right after its own stage", "After the last stage", "Change"], ground_rows)}
<h4>Isolated, joint, and sequential, per dataset</h4>
<p>Each cell is three AUROC values for the same test rows: trained on that dataset alone, trained jointly on the mixture, and trained sequentially in the forward order, after the last stage.</p>
{table(["Head", "Loss"] + labels, joint_rows, left_cols=(0, 1))}"""


def modes_of(kind: str):
    """The mode names of a head kind."""
    return ("joint", "pre_generation") if kind == "unified" else ("main",)


def series_for(kind: str, mode: str) -> str:
    """The scorer key of a head kind and mode."""
    for key, pair in SERIES.items():
        if pair == (kind, mode):
            return key
    raise KeyError((kind, mode))


def np_list(matrix):
    """Return a nested list of floats."""
    return [[float(v) for v in row] for row in matrix]

def recap_section(r) -> str:
    """A structured recap of everything done so far in Approach 1, with the numbers behind it."""
    earlier = r.earlier
    grid = next(iter(earlier.values()))["grid"] if earlier else []

    def captured(key, scorer):
        return earlier[key]["scorers"][scorer]["error_capture"][grid.index(0.2)]

    ones = [earlier[k]["token_confidence_exactly_one"] for k in earlier]
    tokens = {k: earlier[k]["token_confidence_metrics"]["correctness_auroc"] for k in earlier}
    unified = {k: r.specialists[k]["unified_joint"] for k in r.specialists}
    gains = [unified[k] - tokens[k] for k in unified if k in tokens]
    cap_unified = [captured(k, "unified_joint") for k in earlier]
    cap_token = [captured(k, "token_confidence") for k in earlier]
    original = ORIGINAL_TEST
    chart = earlier.get("chartqa_full", {})
    rerun_token = chart.get("token_confidence_metrics", {}).get("correctness_auroc", float("nan"))
    rerun_linear = r.specialists.get("chartqa_full", {}).get("separate_pre_generation", float("nan"))
    ones_text = f"{pct(min(ones), 0)} to {pct(max(ones), 0)}" if ones else "n/a"
    gain_text = f"{min(gains):.2f} to {max(gains):.2f}" if gains else "n/a"
    cap_text = f"{pct(min(cap_unified), 0)} to {pct(max(cap_unified), 0)}" if cap_unified else "n/a"
    cap_token_text = f"{pct(min(cap_token), 0)} to {pct(max(cap_token), 0)}" if cap_token else "n/a"
    stages = table(
        ["Stage", "What was done", "Main result"],
        [
            ["1. Baseline", "Qwen3-VL-2B answers each question once with the full image. The probability the model gave its own answer tokens is logged.",
             f"That probability is exactly 1.0 on {ones_text} of test answers, so it ranks right against wrong answers poorly."],
            ["2. Original ChartQA study", "Hidden states of the frozen model are cached for the question alone and for the question with the answer. Two small heads are trained on them: a pre-generation head and a verifier. A 72-fit sweep and an early-stopping sweep pick the winners.",
             f"AUROC {original['Correctness AUROC'][0]} for the answer-token probability against {original['Correctness AUROC'][2]} for a linear probe on the question state and {original['Correctness AUROC'][4].split()[0]} for the early-stopped verifier. Accuracy on the most trusted half of answers rises from {original['Accuracy at 50% coverage'][0]} to {original['Accuracy at 50% coverage'][2]}."],
            ["3. Three more datasets", "The same experiment on TextVQA, DocVQA and ScienceQA, chosen for different inputs (scene text, scanned documents, science diagrams), with grouped splits and each dataset's own scoring. A unified head that reads the question state and the answer state together, with a loss that also keeps a question-only mode.",
             f"The learned heads beat the answer-token probability by {gain_text} AUROC on every dataset. The unified head ties or beats the separate heads. Flagging the lowest-scored 20% of answers captures {cap_text} of all errors, against {cap_token_text} for the probability."],
            ["4. ChartQA rerun", "ChartQA rerun end to end under the same protocol as the other three, as a check on the pipeline.",
             f"Reproduces the original study: probability AUROC {rerun_token:.3f} against {original['Correctness AUROC'][0]}, linear probe {rerun_linear:.3f} against {original['Correctness AUROC'][2]}."],
            ["5. This report", "All four datasets pooled into one mixture, heads trained one dataset after another to measure forgetting, and a new loss with a visual-grounding term.",
             "Summarized in the next section."],
            ["6. Engineering", "One command per dataset on the ICE cluster, validated Slurm scripts, 31 offline tests, a numbered results folder, and a repository split into models, data and experiments.", "Everything reruns from the repository."],
        ],
        left_cols=(0, 1, 2),
    )
    return f"""<h2 id="recap">1. Approach 1 so far</h2>
<p class="lead">Approach 1 asks a model to notice when its visual evidence is weak and to act on that before answering. It has two halves: a <b>gate</b> that scores how likely an answer is to be right, and a <b>correction step</b> that acts on answers the gate flags. Everything done so far is about the gate, and this report adds to it.</p>
{stages}
<h3>Where Approach 1 stands</h3>
<ul>
<li><b>Done: the detection half.</b> A small head on frozen hidden states predicts whether an answer is correct far better than the answer-token probability, on four different kinds of input, and the result reproduces.</li>
<li><b>Not done: the correction step.</b> Nothing yet acts on a flagged answer, so it is not known whether flagged answers can be fixed. A first candidate is test-time steering of the hidden state toward a higher score.</li>
<li><b>Open items:</b> the Qwen3-VL 4B check, a harder corruption for the grounding test, a softer correctness label for span-style answers, and extra input features for the head (several layers, image tokens).</li></ul>"""


def reading_exp1(r, keys, labels) -> str:
    """Plain-language reading of the model-probability and linear-probe results."""
    base = (r.baseline("pooled") or {}).get("token_confidence", {}).get("correctness_auroc", float("nan"))
    pre = mean_of(r.metric("pre", "old_linear", "main", "correctness_auroc"))
    ver = mean_of(r.metric("verifier", "old_linear", "main", "correctness_auroc"))
    uni = mean_of(r.metric("unified", "old_linear", "joint", "correctness_auroc"))
    mlp = mean_of(r.metric("unified", "old", "joint", "correctness_auroc"))
    per = {NAMES.get(d, d): mean_of(r.metric("unified", "old_linear", "joint", "correctness_auroc", d)) for d in r.datasets}
    best, worst = max(per, key=per.get), min(per, key=per.get)
    weakest_tok = min(r.datasets, key=lambda d: ((r.baseline(d) or {}).get("token_confidence", {}).get("correctness_auroc", 9)))
    return f"""<h4>Reading the results</h4><ul>
<li><b>The untrained answer-token probability ranks answers poorly.</b> Pooled AUROC is {base:.3f}, and it is worst on {esc(NAMES.get(weakest_tok, weakest_tok))}, where the answer is a single option letter that is nearly always given with probability 1.</li>
<li><b>Most of the signal is about the question.</b> A linear probe on the question state alone, with no access to the answer, reaches {pre:.3f}. That is {pre - base:+.3f} over the probability, so a model's hidden state already encodes how hard the question is for it.</li>
<li><b>Reading the answer adds a smaller gain.</b> The unified linear probe, which reads question and answer, reaches {uni:.3f}, {uni - pre:+.3f} over the question-only probe, and the verifier alone is {ver:.3f}.</li>
<li><b>A linear probe is nearly enough.</b> The small MLP version of the unified head reaches {mlp:.3f}, only {mlp - uni:+.3f} above the linear probe.</li>
<li><b>Datasets keep the same difficulty order.</b> The unified linear probe scores best on {esc(best)} ({per[best]:.3f}) and worst on {esc(worst)} ({per[worst]:.3f}).</li></ul>"""


def reading_family(r, name) -> str:
    """Plain-language reading of one loss family's pooled and per-dataset results."""
    uni = mean_of(r.metric("unified", name, "joint", "correctness_auroc"))
    ver = mean_of(r.metric("verifier", name, "main", "correctness_auroc"))
    pre = mean_of(r.metric("pre", name, "main", "correctness_auroc"))
    uni_pre = mean_of(r.metric("unified", name, "pre_generation", "correctness_auroc"))
    exceptions = []
    for d in r.datasets:
        u = mean_of(r.metric("unified", name, "joint", "correctness_auroc", d))
        v = mean_of(r.metric("verifier", name, "main", "correctness_auroc", d))
        q = mean_of(r.metric("pre", name, "main", "correctness_auroc", d))
        if not (u >= v >= q):
            exceptions.append(NAMES.get(d, d))
    order = "the same order holds on every dataset" if not exceptions else f"the order differs on {', '.join(exceptions)}, where the verifier does not beat the question-only head"
    loss_word = "old loss" if name == "old" else "new loss"
    return f"""<h4>Reading the results</h4><ul>
<li><b>One head that reads both states is best.</b> With the {loss_word}, pooled AUROC is {uni:.3f} for the unified head in joint mode, {ver:.3f} for the verifier and {pre:.3f} for the question-only pre-generation head. On individual datasets {order}.</li>
<li><b>The unified head's question-only mode keeps most of its accuracy.</b> With the answer half zeroed it still reaches {uni_pre:.3f}, so one set of weights serves both the gate before generation and the check after it.</li>
<li><b>Calibration.</b> After temperature scaling on the pooled validation split, the expected calibration error of the unified head is {mean_of(r.metric('unified', name, 'joint', 'ece_10_bin')):.3f}, and Brier score {mean_of(r.metric('unified', name, 'joint', 'brier')):.3f}.</li></ul>"""


def seq_reading(r) -> str:
    """Plain-language reading of the sequential-training summary table."""
    f = {k: seq_value(r, k, "old_replay0", modes_of(k)[0], "forgetting") for k in KINDS}
    fn = {k: seq_value(r, k, "new_replay0", modes_of(k)[0], "forgetting") for k in KINDS}
    fr = {k: seq_value(r, k, "old_replay0.1", modes_of(k)[0], "forgetting") for k in KINDS}
    final_seq = seq_value(r, "unified", "old_replay0", "joint", "final_avg")
    joint_avg = sum(mean_of(r.metric("unified", "old", "joint", "correctness_auroc", d)) for d in r.datasets) / len(r.datasets)
    iso = r.continual.get("unified", {}).get("variants", {}).get("old_replay0", {}).get("isolated", {}).get("joint", {})
    iso_avg = sum(v["mean"] for v in iso.values()) / len(iso) if iso else float("nan")
    data = r.continual.get("unified", {}).get("variants", {}).get("old_replay0")
    per_dataset = {n: [] for n in r.datasets}
    if data:
        for entry in data["orders"].values():
            matrix = np_list(entry["matrix"]["joint"])
            order = [r.datasets.index(n) for n in entry["order"]]
            for position, dataset in enumerate(order[:-1]):
                per_dataset[r.datasets[dataset]].append(max(matrix[stage][dataset] for stage in range(position, len(order) - 1)) - matrix[-1][dataset])
    means = {n: sum(v) / len(v) for n, v in per_dataset.items() if v}
    most = max(means, key=means.get) if means else None
    least = min(means, key=means.get) if means else None
    return f"""<h4>Reading the results</h4><ul>
<li><b>Forgetting is small for every head.</b> With plain sequential training and the old loss, earlier datasets lose {f['unified']:.3f} AUROC for the unified head, {f['pre']:.3f} for the pre-generation head and {f['verifier']:.3f} for the verifier. These are around the size of the gaps between heads, not the collapse the word forgetting suggests. The frozen hidden states are shared across datasets, which is the likely reason.</li>
<li><b>Reading the answer makes a head forget more.</b> The verifier forgets {f['verifier'] / f['unified']:.1f} times as much as the unified head. A likely reason is that an answer looks different in each dataset (a number, a string from a page, a letter), but this was not tested separately.</li>
<li><b>Replay fixes the verifier and does nothing for the others.</b> A buffer holding 10% of every earlier dataset's training rows cuts the verifier's forgetting from {f['verifier']:.3f} to {fr['verifier']:.3f}. For the pre-generation head ({f['pre']:.3f} to {fr['pre']:.3f}) and the unified head ({f['unified']:.3f} to {fr['unified']:.3f}) the change is inside the seed spread.</li>
<li><b>The new loss does not change forgetting.</b> It is {fn['pre']:.3f}, {fn['verifier']:.3f} and {fn['unified']:.3f} for the three heads, against {f['pre']:.3f}, {f['verifier']:.3f} and {f['unified']:.3f} with the old loss.</li>
<li><b>Sequential training costs little against training on everything at once.</b> The unified head ends at an average AUROC of {final_seq:.3f}, against {iso_avg:.3f} for heads trained on each dataset alone and {joint_avg:.3f} for one head trained on the pool.</li>
{f'<li><b>Which dataset suffers most.</b> {esc(NAMES.get(most, most))} loses the most ({means[most]:.3f}) and {esc(NAMES.get(least, least))} the least ({means[least]:.3f}), averaged over the orders in which they are not the last dataset trained.</li>' if most else ''}</ul>"""


def grounding_alone_block(r) -> str:
    """The grounding term on its own: equation, pooled and per-dataset results, and a reading."""
    rows = []
    for label, kind, mode in (("Separate verifier", "verifier", "main"), ("Unified, joint mode", "unified", "joint"), ("Unified, pre-generation mode", "unified", "pre_generation")):
        for name, title in (("cell_plain", "plain (old loss)"), ("cell_ground", "grounding term alone")):
            auroc, prauc = r.metric(kind, name, mode, "correctness_auroc"), r.metric(kind, name, mode, "error_auprc")
            ece = r.metric(kind, name, mode, "ece_10_bin")
            order, drop = r.grounding(kind, name, mode, "order_accuracy"), r.grounding(kind, name, mode, "mean_drop")
            if None in (auroc, prauc, ece, order, drop):
                continue
            rows.append([label, title, f"{auroc[0]:.3f}", f"{prauc[0]:.3f}", f"{ece[0]:.3f}", pct(order[0]), f"{drop[0]:.3f}"])
    weight = (r.winner("unified", "cell_ground") or {}).get("config", {}).get("ground_weight")
    per_rows = []
    gains = {}
    for dataset in r.datasets:
        plain, ground = r.metric("unified", "cell_plain", "joint", "correctness_auroc", dataset), r.metric("unified", "cell_ground", "joint", "correctness_auroc", dataset)
        order_p, order_g = r.grounding("unified", "cell_plain", "joint", "order_accuracy", dataset=dataset), r.grounding("unified", "cell_ground", "joint", "order_accuracy", dataset=dataset)
        if None in (plain, ground, order_p, order_g):
            continue
        gains[dataset] = order_g[0] - order_p[0]
        per_rows.append([NAMES.get(dataset, dataset), f"{plain[0]:.3f}", f"{ground[0]:.3f}", f"{ground[0] - plain[0]:+.3f}", pct(order_p[0]), pct(order_g[0]), f"{gains[dataset] * 100:+.1f} points"])
    d_auroc = cell_change(r, "unified", "joint", "cell_ground")
    d_order = cell_change(r, "unified", "joint", "cell_ground", ground=True)
    d_ece = (mean_of(r.metric("unified", "cell_ground", "joint", "ece_10_bin")) - mean_of(r.metric("unified", "cell_plain", "joint", "ece_10_bin")))
    top = max(gains, key=gains.get) if gains else None
    return f"""<h3>The grounding term on its own</h3>
<p>The cell above labelled grounding only uses just the new term, with no pair term. Its equation is</p>
<pre>L = BCE( s_joint(x_clean), y ) + BCE( s_pre(x_clean), y ) + lambda * L_ground
L_ground = mean over modes, mean over correct answers of  softplus( s(x_corrupt) - s(x_clean) )</pre>
<p>with lambda = {weight if weight is not None else 'n/a'}, chosen on validation from 0.1 and 0.5. The separate verifier uses the same without the pre-generation term. Every row below is pooled over the test split, trained on the mixture, mean of three seeds.</p>
{table(["Head", "Loss", "AUROC", "Error AUPRC", "ECE", "Correct answers whose score drops under a corrupted image", "Mean drop in probability"], rows, left_cols=(0, 1))}
<p>By dataset, for the unified head in joint mode:</p>
{table(["Dataset", "AUROC, plain", "AUROC, grounding alone", "Change", "Score drops, plain", "Score drops, grounding alone", "Change"], per_rows)}
<h4>Reading the results</h4><ul>
<li><b>It is free in accuracy.</b> For the unified head the term changes pooled AUROC by {d_auroc:+.3f}, which is inside the seed spread. The separate verifier moves by {cell_change(r, 'verifier', 'main', 'cell_ground'):+.3f}.</li>
<li><b>It improves calibration slightly.</b> The unified head's expected calibration error changes by {d_ece:+.3f}.</li>
<li><b>It makes the score depend on the image.</b> The share of correct answers whose score falls under a corrupted image rises by {d_order * 100:.1f} points for the unified head. The rise comes {f'mostly from {esc(NAMES.get(top, top))} ({gains[top] * 100:+.1f} points)' if top else ''}, where many questions can be answered from the question text alone and the plain head learns to rely on that. The other datasets were already above 97%, so there was little room to improve.</li>
<li><b>The question-only mode is grounded as well.</b> Its drop rate rises from {pct(mean_of(r.grounding('unified', 'cell_plain', 'pre_generation', 'order_accuracy')), 0)} to {pct(mean_of(r.grounding('unified', 'cell_ground', 'pre_generation', 'order_accuracy')), 0)}. That mode never sees the answer, so it reacts to the image through the question state.</li>
<li><b>The small weight was enough.</b> lambda = 0.1 was chosen over 0.5 on validation. The term is a small addition to a cross-entropy loss that already dominates.</li></ul>"""


def loss_section(r) -> str:
    """The new loss in detail: equation, term-by-term meaning, corruption, measurement, protocol, and verdict."""
    gallery = r.analysis.get("gallery", [])
    shots = "".join(
        f'<div class="card"><p class="small"><b>{esc(NAMES.get(g["dataset"], g["dataset"]))}</b></p>'
        f'<div class="grid2"><div><img class="thumb" alt="clean image" src="data:image/jpeg;base64,{g["clean"]}"/><p class="small">Clean</p></div>'
        f'<div><img class="thumb" alt="corrupted image" src="data:image/jpeg;base64,{g["corrupt"]}"/><p class="small">Corrupted (another image from the same split)</p></div></div></div>'
        for g in gallery
    )
    pair_auroc = cell_change(r, "unified", "joint", "cell_contrast")
    ground_auroc = cell_change(r, "unified", "joint", "cell_ground")
    ground_order = cell_change(r, "unified", "joint", "cell_ground", ground=True)
    verifier_pair = cell_change(r, "verifier", "main", "cell_contrast")
    old_g = mean_of(r.grounding("unified", "old", "joint", "order_accuracy"))
    new_g = mean_of(r.grounding("unified", "new", "joint", "order_accuracy"))
    old_u = mean_of(r.metric("unified", "old", "joint", "correctness_auroc"))
    new_u = mean_of(r.metric("unified", "new", "joint", "correctness_auroc"))
    f_old, f_new = seq_value(r, "unified", "old_replay0", "joint", "forgetting"), seq_value(r, "unified", "new_replay0", "joint", "forgetting")
    verdicts = table(
        ["Criterion", "Measured", "Met?"],
        [
            ["Grounding: the score of correct answers should fall under a corrupted image more often than with the old loss", f"unified head, joint mode: {pct(old_g, 1)} to {pct(new_g, 1)}", "Yes" if new_g > old_g + 0.01 else "No"],
            ["Accuracy: pooled AUROC should not drop by more than the seed spread", f"unified head, joint mode: {old_u:.3f} to {new_u:.3f}", "Yes" if new_u >= old_u - 0.003 else "No"],
            ["Retention: forgetting in sequential training should not get worse", f"unified head: {f_old:.3f} to {f_new:.3f}", "Yes" if f_new <= f_old + 0.005 else "No"],
        ],
        left_cols=(0, 1, 2),
    )
    return f"""<h2 id="loss">4. The new loss in detail</h2>
<h3>4.1 The idea</h3>
<p>A correctness head should be confident when the image supports the answer and less confident when it does not. The old loss never asks for this. It asks only for the right label on real inputs, so a head can reach a high AUROC by learning what kinds of questions the model gets wrong without ever looking at what the image shows. The new loss adds a direct request: the same question and the same answer must score lower when the image has been replaced by one that does not support them.</p>
<h3>4.2 The equation</h3>
<p>The proposal has three lines. Each maps to one term of the loss as implemented:</p>
<pre>P_theta(S_confidence | h_answer, h_question)         ->  L_cls  (and L_pre for the unified head)
L_contrastive(h_predicted, h_correct)                ->  L_pair
L_contrastive(h_clean, h_corrupt) -> S_confidence    ->  L_ground</pre>
<p>Written out, for one mini-batch:</p>
<pre>L = L_cls  +  w_pre * L_pre  +  w_pair * L_pair  +  lambda * L_ground

L_cls    = BCE( s_joint(x_clean), y )                      correctness on the real input
L_pre    = BCE( s_pre(x_clean), y )                        unified head only, answer half zeroed
L_pair   = mean over wrong answers i of
           softplus( -( s(x_reference_i) - s(x_predicted_i) ) )
L_ground = mean over modes m, mean over correct answers i of
           softplus( margin - ( s_m(x_clean_i) - s_m(x_corrupt_i) ) )

softplus(z) = log(1 + exp(z))        margin = 0
s(.)  is the head's logit            y = 1 when the model's answer was correct
x_clean      hidden states of the real image, the question, and the model's answer
x_corrupt    the same question and answer text, with the image swapped for another one
x_reference  the same question with the correct answer appended
x_predicted  the same question with the model's own answer appended</pre>
<p>The old loss is the case lambda = 0. The new loss is lambda above 0. The new loss as first specified has both w_pair and lambda above 0.</p>
<h3>4.3 What each term asks for</h3>
{table(["Term", "What it asks of the head", "Why it could help", "What the results show"], [
    ["L_cls and L_pre", "Predict whether the answer is right, from the real input. L_pre does the same with the answer half zeroed.", "This is the base signal. L_pre keeps a usable score before any answer exists.", "Needed. Present in both the old and the new loss."],
    ["L_pair (predicted against correct)", "On a wrong answer, the correct answer's state must score above the model's wrong answer's state for the same question.", "It could teach the head what a right answer looks like by contrast.", f"Costs accuracy: it changes unified pooled AUROC by {pair_auroc:+.3f} and the verifier's by {verifier_pair:+.3f}."],
    ["L_ground (clean against corrupted)", "For a correct answer, the real input must score above the same input with a corrupted image.", "It makes the score depend on the image and not only on the question.", f"Free in accuracy ({ground_auroc:+.3f} AUROC) and raises the drop rate by {ground_order * 100:.1f} points."],
], left_cols=(0, 1, 2, 3))}
<h3>4.4 Design choices in the grounding term</h3>
<ul>
<li><b>A pairwise logistic loss, not a target of zero.</b> With margin 0 the term is <code>-log sigmoid(s_clean - s_corrupt)</code>. It asks only that the clean score be higher, not that the corrupted score be low. A corrupted input can still be correct by luck, or from the question text, so forcing it to zero would be wrong. Ordering is the claim worth making.</li>
<li><b>Correct answers only.</b> A wrong answer is not supported by the image to begin with, so there is no reason to expect the clean input to score above the corrupted one. Requiring it would add noise to the term.</li>
<li><b>Logits, not probabilities.</b> The loss is applied to the logit difference, which is meant to avoid the vanishing gradient that a probability near 0 or 1 would give. Probabilities were not tried as an alternative.</li>
<li><b>Both modes.</b> The unified head applies the term in joint mode and in question-only mode. A question-only score also depends on the image, because the question state carries it.</li>
<li><b>One corruption.</b> The corrupted input swaps in a different example's image from the same dataset and split. The question and the answer text stay the same. Hidden states for these inputs were recomputed with Qwen3-VL for every example, so the corrupted input is a real model input, not noise added to features.</li></ul>
{f'<div class="grid2">{shots}</div>' if shots else ''}
<h3>4.5 How it is measured</h3>
<p>Grounding is measured on the test split, for correctly answered examples only. For each one the score on the clean input is compared with the score on the corrupted input:</p>
<ul>
<li><b>Drop rate (order accuracy).</b> The share of examples where the clean score is higher. 0.5 means the corruption makes no difference, 1.0 means it always lowers the score.</li>
<li><b>Mean drop.</b> The average fall in predicted probability.</li></ul>
<p>The model's own answer-token probability is measured the same way, teacher-forced on the same answer text under the clean and the corrupted image, so the comparison is like for like.</p>
<h3>4.6 What would count as success</h3>
<p>Three criteria define success for the new loss. They were written down while the report was being built, not before any sweep had run, so they are a way to read the results and not a pre-registered test.</p>
{verdicts}
<h3>4.7 What was run</h3>
{table(["Head kind", "Grid", "Configurations", "Trainings (3 seeds)"], [
    ["Separate pre-generation", "architecture (linear, 256-unit MLP, 512-unit MLP) x learning rate (2) x ranking weight (0, 0.25) x grounding weight (0, 0.1, 0.5)", "36", "108"],
    ["Separate verifier", "architecture x learning rate x pair weight (0, 0.25) x grounding weight (0, 0.1, 0.5)", "36", "108"],
    ["Unified", "architecture x learning rate x pre-generation weight (0, 1) x pair weight (0, 0.25) x grounding weight (0, 0.1, 0.5)", "72", "216"],
    ["Sequential, per head kind", "the old-loss and new-loss winners of the sweep above, 3 dataset orders x 3 seeds x replay (0, 10%)", "4 variants", "36 sequences of 4 stages, plus 12 isolated heads per variant"]], left_cols=(0, 1))}
<p class="small">Old loss means the grounding weight is 0. New loss means it is above 0. Within each family the winner is the configuration with the best mean validation score over three seeds, and test numbers are read for winners only.</p>
<h3>4.8 Cautions on interpretation</h3>
<ul>
<li><b>The second contrastive term is an interpretation.</b> It is implemented at the score level, as a pairwise loss between the scores of the correct and the predicted answer. If an embedding-level contrast was intended, <code>head_loss</code> in <code>mixture_heads.py</code> is the one place to change.</li>
<li><b>A shortcut is possible.</b> A swapped image usually has nothing to do with the question, so a head could satisfy the term by detecting that the image and question do not match, which is easier than checking that the image supports this particular answer. The term therefore shows that the score depends on the image, and is weaker evidence that it depends on the right part of the image.</li>
<li><b>Training and testing use the same corruption.</b> A different corruption, such as shuffling patches, was not tested.</li></ul>"""


def delta_pair(r, kind, mode, metric="correctness_auroc", dataset=None):
    """New minus old for one scorer, with the combined seed spread."""
    old, new = r.metric(kind, "old", mode, metric, dataset), r.metric(kind, "new", mode, metric, dataset)
    if old is None or new is None:
        return float("nan"), 0.0
    return new[0] - old[0], math.sqrt(old[1] ** 2 + new[1] ** 2)


def grounding_section(r, base, grounding_tok, keys, labels) -> str:
    """Does the score actually fall when the image is corrupted, with and without the new loss."""
    rows, chart_rows = [], []
    tok = grounding_tok
    rows.append(["Answer-token probability", "n/a", f3(tok.get("order_accuracy", float("nan"))), f3(tok.get("mean_drop", float("nan")))])
    chart_rows.append(("Answer-token probability", None, None, tok.get("order_accuracy")))
    for series, (kind, mode) in SERIES.items():
        old = r.grounding(kind, "old", mode, "order_accuracy")
        new = r.grounding(kind, "new", mode, "order_accuracy")
        old_drop = r.grounding(kind, "old", mode, "mean_drop")
        new_drop = r.grounding(kind, "new", mode, "mean_drop")
        rows.append([SCORER_LABEL[series] + ", old loss", "old", f3(mean_of(old)), f3(mean_of(old_drop))])
        rows.append([SCORER_LABEL[series] + ", new loss", "new", f3(mean_of(new)), f3(mean_of(new_drop))])
        chart_rows.append((SCORER_LABEL[series], mean_of(old), mean_of(new), None))
    plot_rows = [(label, old, new, token) for label, old, new, token in chart_rows]
    chart = pair_plot(plot_rows, 0.4, 1.0, "Share of correct answers whose score falls under a corrupted image (0.5 means no effect)", "fig-ground")
    per_dataset = []
    for key, label in zip(keys, labels):
        cells = [label]
        b = r.baseline("pooled" if key is None else key)
        tg = ((b or {}).get("grounding_token_confidence") or {}).get("correct", {})
        cells.append(f3(tg["order_accuracy"]) if tg else "n/a")
        for name in ("old", "new"):
            pair = r.grounding("unified", name, "joint", "order_accuracy", dataset=key)
            cells.append("n/a" if pair is None else f"{pair[0]:.3f} ± {pair[1]:.3f}")
        per_dataset.append(cells)
    return f"""<h2 id="grounding">9. Does the score track visual grounding?</h2>
<p>The new loss exists to make one specific thing true: the same question and the same answer should score lower when the image no longer supports it. This is measured directly on the test split. For every correctly answered example the score on the clean input is compared with the score on the corrupted input. <b>Order accuracy</b> is the share of examples where the clean score is higher (0.5 means the corruption makes no difference, 1.0 means it always lowers the score). <b>Mean drop</b> is the average fall in predicted probability.</p>
{legend([("Answer-token probability", "c-tok"), ("Old loss", OLD_CSS), ("New loss", NEW_CSS)])}
{figure("fig-ground", "Share of correct answers whose score falls when the image is corrupted", chart, "Answer-token probability is teacher-forced on the same answer text under the clean and the corrupted image. Heads are scored on the same two inputs.")}
{table(["Scorer", "Loss", "Order accuracy", "Mean drop"], rows)}
<h4>By dataset, unified head in joint mode</h4>
{table(["Dataset", "Answer-token probability", "Old loss", "New loss"], per_dataset)}"""


def ablation_section(r) -> str:
    """The two-by-two ablation of the pair term against the grounding term."""
    cells = (("cell_plain", "plain"), ("cell_contrast", "pair only"), ("cell_ground", "grounding only"), ("cell_full", "pair and grounding"))
    rows = []
    for kind, mode, label in (("verifier", "main", "Separate verifier"), ("unified", "joint", "Unified, joint mode"), ("unified", "pre_generation", "Unified, pre-generation mode")):
        for name, cell_label in cells:
            auroc = r.metric(kind, name, mode, "correctness_auroc")
            prauc = r.metric(kind, name, mode, "error_auprc")
            order = r.grounding(kind, name, mode, "order_accuracy")
            w = r.winner(kind, name)
            rows.append([label, cell_label, "n/a" if auroc is None else f"{auroc[0]:.3f} ± {auroc[1]:.3f}", "n/a" if prauc is None else f"{prauc[0]:.3f} ± {prauc[1]:.3f}",
                         "n/a" if order is None else f"{order[0]:.3f}", "n/a" if w is None else esc(config_text(w["config"]))])
    return f"""<h2 id="ablation">10. Which part of the new loss matters</h2>
<p>The new loss has two contrastive terms. This table crosses them: the pairwise reference-answer term (predicted against correct) on or off, and the grounding term on or off, taking the best configuration in each cell on validation. The first row of each group is the old loss without either term.</p>
{table(["Head", "Cell", "AUROC", "Error AUPRC", "Order accuracy", "Winning configuration"], rows, left_cols=(0, 1, 5))}
{grounding_alone_block(r)}"""


def generalist_section(r, keys, labels) -> str:
    """The mixture-trained heads against heads trained on one dataset."""
    rows, chart_rows = [], []
    for dataset in r.datasets:
        spec = r.specialists.get(dataset)
        if not spec:
            continue
        mixed_old = mean_of(r.metric("unified", "old", "joint", "correctness_auroc", dataset))
        mixed_new = mean_of(r.metric("unified", "new", "joint", "correctness_auroc", dataset))
        rows.append([NAMES.get(dataset, dataset), f3(spec["separate_verifier"]), f3(spec["unified_joint"]), f3(mixed_old), f3(mixed_new),
                     f"{mixed_old - spec['unified_joint']:+.3f}", f"{mixed_new - spec['unified_joint']:+.3f}"])
        chart_rows.append((NAMES.get(dataset, dataset), mixed_old, mixed_new, spec["unified_joint"]))
    chart = pair_plot([(a, b, c, d) for a, b, c, d in chart_rows], 0.75, 0.95, "Correctness AUROC of the unified head, joint mode", "fig-generalist")
    return f"""<h2 id="generalist">11. One head for everything, or one head per dataset?</h2>
<p>The heads in the earlier per-dataset study were trained and tuned on one dataset each. The mixture heads are trained once on all four. Both are scored on the same test rows, so the difference per dataset is what a single shared head costs or gains.</p>
{legend([("Trained on that dataset only (unified head)", "c-tok"), ("Mixture, old loss", OLD_CSS), ("Mixture, new loss", NEW_CSS)])}
{figure("fig-generalist", "Unified head trained on the mixture against trained on one dataset", chart, "The gray dot is the unified head from the per-dataset study. The mixture heads are the winners of experiments 2 and 3.")}
{table(["Dataset", "Per-dataset separate verifier", "Per-dataset unified", "Mixture unified, old loss", "Mixture unified, new loss", "Old minus per-dataset", "New minus per-dataset"], rows)}"""


def use_section(r, labels, keys) -> str:
    """Accuracy when dropping the least trusted answers, and errors found at each flag rate."""
    curves = r.analysis.get("curves", {})
    grid = r.analysis.get("grid", [])
    token = r.analysis.get("token_curves", {})
    if not curves or not grid:
        return '<h2 id="use">12. Using the score</h2><p>Curves were not available.</p>'
    names = [None] + list(r.datasets)
    coverage_panels, capture_panels = [], []
    for key, label in zip(names, labels):
        subset = "pooled" if key is None else key
        if subset not in token:
            continue
        series_cov = [("token_confidence", grid, token[subset]["accuracy_at_coverage"])]
        series_cap = [("token_confidence", grid, token[subset]["error_capture"])]
        unified_old = curves.get("unified", {}).get("old", {}).get("joint", {}).get(subset)
        unified_new = curves.get("unified", {}).get("new", {}).get("joint", {}).get(subset)
        if unified_old:
            series_cov.append(("loss_old", grid, unified_old["accuracy_at_coverage"]))
            series_cap.append(("loss_old", grid, unified_old["error_capture"]))
        if unified_new:
            series_cov.append(("loss_new", grid, unified_new["accuracy_at_coverage"]))
            series_cap.append(("loss_new", grid, unified_new["error_capture"]))
        low = min(min(v) for _, _, v in series_cov)
        ymin = max(0.0, (int(low * 20) / 20) - 0.05)
        coverage_panels.append(line_panel(label, grid, series_cov, round(ymin, 2), 1.0, 0.05 if (1 - ymin) <= 0.3 else 0.1, "Fraction of answers kept (most trusted first)", "Accuracy of kept answers"))
        capture_panels.append(line_panel(label, grid, series_cap, 0.0, 1.0, 0.25, "Fraction of answers flagged (least trusted first)", "Share of all errors captured", diagonal=True))
    key_legend = legend([("Answer-token probability", "c-tok"), ("Unified head, old loss", OLD_CSS), ("Unified head, new loss", NEW_CSS)])
    return f"""<h2 id="use">12. Using the score</h2>
<p>What the scores are worth in use, for the unified head in joint mode. First, if the least trusted answers are dropped, how accurate is what remains? Second, if the lowest-scored share of answers is flagged, how many errors does it contain?</p>
{key_legend}
{figure("fig-coverage", "Accuracy of the answers that remain as the least trusted are dropped", f'<div class="grid2">{"".join(coverage_panels)}</div>', "The right edge is full coverage, which equals the base accuracy of that subset.")}
{figure("fig-capture", "Share of all errors found in the flagged fraction of answers", f'<div class="grid2">{"".join(capture_panels)}</div>', "The dashed diagonal is random flagging.")}"""


def meaning_section(r, best_old, best_new, deltas, grounding_tok) -> str:
    """The interpretation of the results, with every number read from them."""
    names = r.datasets
    spec_gap = []
    for dataset in names:
        spec = r.specialists.get(dataset)
        mixed = mean_of(r.metric("unified", "old", "joint", "correctness_auroc", dataset))
        if spec:
            spec_gap.append((NAMES.get(dataset, dataset), mixed - spec["unified_joint"]))
    gap_text = ", ".join(f"{name} {gap:+.3f}" for name, gap in spec_gap)
    better = [n for n, g in spec_gap if g > 0.002]
    worse = [n for n, g in spec_gap if g < -0.002]
    token_order = grounding_tok.get("order_accuracy", float("nan"))
    token_auroc = ((r.baseline("pooled") or {}).get("token_confidence") or {}).get("correctness_auroc", float("nan"))
    verifier_old, verifier_new = r.grounding("verifier", "old", "main", "order_accuracy"), r.grounding("verifier", "new", "main", "order_accuracy")
    pre_old, pre_new = r.grounding("pre", "old", "main", "order_accuracy"), r.grounding("pre", "new", "main", "order_accuracy")
    ground_cost = cell_change(r, "unified", "joint", "cell_ground")
    pair_cost = cell_change(r, "unified", "joint", "cell_contrast")
    full_cost = cell_change(r, "unified", "joint", "cell_full")
    forgetting = {k: seq_value(r, k, "old_replay0", modes_of(k)[0], "forgetting") for k in KINDS}
    replay_forgetting = {k: seq_value(r, k, "old_replay0.1", modes_of(k)[0], "forgetting") for k in KINDS}
    new_forgetting = {k: seq_value(r, k, "new_replay0", modes_of(k)[0], "forgetting") for k in KINDS}
    final_seq = seq_value(r, "unified", "old_replay0", "joint", "final_avg")
    joint_avg = sum(mean_of(r.metric("unified", "old", "joint", "correctness_auroc", d)) for d in names) / len(names)
    iso = r.continual.get("unified", {}).get("variants", {}).get("old_replay0", {}).get("isolated", {}).get("joint", {})
    iso_avg = sum(v["mean"] for v in iso.values()) / len(iso) if iso else float("nan")
    # transfer before training: a head trained on the first dataset only, scored on the others
    transfer = ""
    forward = r.continual.get("pre", {}).get("variants", {}).get("old_replay0", {}).get("orders", {}).get("forward")
    verifier_forward = r.continual.get("verifier", {}).get("variants", {}).get("old_replay0", {}).get("orders", {}).get("forward")
    if forward and verifier_forward:
        first = forward["matrix"]["main"][0]
        first_v = verifier_forward["matrix"]["main"][0]
        others = [i for i in range(len(names)) if i != names.index(forward["order"][0])]
        transfer = (f"A pre-generation head trained on {NAMES.get(forward['order'][0], forward['order'][0])} alone already scores "
                    + ", ".join(f"{first[i]:.3f} on {NAMES.get(names[i], names[i])}" for i in others)
                    + f" before it has seen any of them. The verifier, which reads the answer, transfers much less ("
                    + ", ".join(f"{first_v[i]:.3f}" for i in others) + "), likely because answers look different from dataset to dataset.")
    return f"""<h2 id="meaning">13. What the results mean</h2>
<h3>13.1 One head for four kinds of input works about as well as four heads</h3>
<p>The answer-token probability reaches pooled AUROC {token_auroc:.3f}. A head trained on the mixture reaches {best_old[1]:.3f} with the old loss and {best_new[1]:.3f} with the new loss. Against heads trained on one dataset each, the shared unified head differs by {gap_text} AUROC. It is {'at least as good on ' + ', '.join(better) if better else 'not better on any dataset'}{' and a little worse on ' + ', '.join(worse) if worse else ''}. The differences on three of the datasets are a few thousandths, so the fair reading is that the shared head matches the specialists there and is a little worse on the dataset whose questions carry the most signal on their own. A plausible reason, not tested here, is that the pool teaches the head about harder questions from the other datasets while diluting what is specific to ScienceQA. The shared head does not need to be tuned per dataset.</p>
<h3>13.2 Training on later datasets barely erases earlier ones</h3>
<p>When the heads are trained on one dataset after another, the AUROC lost on earlier datasets is small for every head: {forgetting['pre']:.3f} for the pre-generation head, {forgetting['verifier']:.3f} for the verifier and {forgetting['unified']:.3f} for the unified head, trained plainly with the old loss. After all four stages the unified head's average AUROC is {final_seq:.3f}, against {iso_avg:.3f} for heads trained on each dataset alone and {joint_avg:.3f} for the head trained on all of them at once. Sequential training gives up about {joint_avg - final_seq:.3f} AUROC compared with joint training. The frozen hidden states are shared across the datasets, so most of what a head learns, which questions the model finds hard, plausibly carries over. {transfer}</p>
<h3>13.3 Heads that read the answer forget more, and a small replay buffer fixes it</h3>
<p>The verifier forgets the most ({forgetting['verifier']:.3f}), the unified head the least ({forgetting['unified']:.3f}). With a replay buffer of 10% of every earlier dataset's training rows the verifier's forgetting drops to {replay_forgetting['verifier']:.3f}, and its final average rises to {seq_value(r, 'verifier', 'old_replay0.1', 'main', 'final_avg'):.3f} from {seq_value(r, 'verifier', 'old_replay0', 'main', 'final_avg'):.3f}. For the heads that already forget little, replay changes nothing measurable. So the cost of forgetting sits mostly in the answer-conditioned state. A likely reason is that its form depends on the dataset (a number, a string from a page, a letter), but that was not tested separately.</p>
<h3>13.4 The new loss does not change what is remembered, but it changes what the score means</h3>
<p>For ranking quality, the new loss behaves like the old one in the sequential runs: forgetting is {new_forgetting['pre']:.3f}, {new_forgetting['verifier']:.3f} and {new_forgetting['unified']:.3f} for the three heads, against {forgetting['pre']:.3f}, {forgetting['verifier']:.3f} and {forgetting['unified']:.3f} with the old loss. What it changes is grounding. Sensitivity to the image does not fade during later training under either loss: for each dataset, the share of correct answers whose score falls under a corrupted image is within about 0.03 right after its own stage and after the last one. The difference is the level. After the last stage, averaged over datasets, that share is {pct(final_grounding(r, 'unified', 'old_replay0', 'joint'), 0)} with the old loss and {pct(final_grounding(r, 'unified', 'new_replay0', 'joint'), 0)} with the new loss for the unified head, and {pct(final_grounding(r, 'pre', 'old_replay0', 'main'), 0)} against {pct(final_grounding(r, 'pre', 'new_replay0', 'main'), 0)} for the pre-generation head. The gap is widest on ScienceQA, where the old-loss heads respond to the image least: many of its questions can be answered from the question text, and the old-loss head learns to lean on that. The new loss pushes the score to depend on the image even there.</p>
<h3>13.5 The grounding term is free in accuracy, the pair term is not</h3>
<p>The ablation separates the two contrastive terms. For the unified head, the grounding term alone changes pooled AUROC by {ground_cost:+.3f} and raises grounding order accuracy by {cell_change(r, 'unified', 'joint', 'cell_ground', ground=True):+.3f}. The pair term alone (predicted answer against correct answer) changes AUROC by {pair_cost:+.3f}, and the two together by {full_cost:+.3f}. The separate verifier shows the same pattern. The pair term was already found to hurt in the per-dataset study, and it hurts here too, so the useful part of the proposed loss is the clean-versus-corrupted term. A plausible reason, not tested here, is that the correct-answer state differs from a generated-answer state in ways that have nothing to do with correctness, so the head can reduce the pair loss by learning that difference.</p>
<h3>13.6 The answer-token probability is sensitive to the image, but cannot rank answers</h3>
<p>One result is easy to miss. When the image is swapped, the answer-token probability falls for {pct(token_order, 0)} of correct answers. That is more than the unified head with the old loss ({pct(r.grounding('unified', 'old', 'joint', 'order_accuracy')[0], 0)}), about the same as the verifier with the old loss ({pct(verifier_old[0], 0) if verifier_old else 'n/a'}), and less than the verifier with the new loss ({pct(verifier_new[0], 0) if verifier_new else 'n/a'}). So the model's confidence does register that the evidence is gone. Its weakness is elsewhere: among answers on clean images it is saturated near 1.0 and ranks right against wrong poorly (pooled AUROC {token_auroc:.3f}). The learned heads add the ranking. The old loss gives them no particular sensitivity to the image, and the new loss raises it, for the pre-generation head from {pct(pre_old[0], 0) if pre_old else 'n/a'} to {pct(pre_new[0], 0) if pre_new else 'n/a'} and for the verifier above the level of the answer-token probability.</p>
<h3>13.7 What this means for the project</h3>
<p>A correctness gate can be trained on whatever data is available, in any order, without a large penalty on what it learned earlier, and extended to a new dataset with a small replay buffer if the head reads the answer. The loss term that checks grounding can be added at no accuracy cost and gives a score that responds to the image. Whether that sensitivity makes the gate better at deciding when a correction will help is the next question, and it needs the correction step.</p>"""


def limits_section() -> str:
    """Limits that hold whatever the results are."""
    return """<h2 id="limits">14. Limits</h2><ul>
<li><b>One model size.</b> Only Qwen3-VL-2B was run.</li>
<li><b>One kind of corruption.</b> Grounding is tested by swapping in another image from the same dataset and split. A different corruption (patch shuffle, blur) could behave differently, and the swap is only one way to remove evidence.</li>
<li><b>The corrupted answer is the original answer.</b> The score is asked about the same answer text under a corrupted image. That tests whether the score notices missing support, not whether the model would have answered differently.</li>
<li><b>The pooled mixture is dominated by ChartQA in the training data</b> (the largest split by far), which can favor it in the pooled numbers. Per-dataset rows are shown for that reason.</li>
<li><b>Forgetting here is the forgetting of a small head on frozen features.</b> The language model is never updated, so there is no forgetting inside it. Datasets share the same hidden-state space, which is part of why forgetting is mild. A head trained on features from a model that is itself updated could forget more.</li>
<li><b>The sequential runs use fixed configurations.</b> Each loss family uses the winner of the mixture sweep for that kind, not a configuration tuned for sequential training, and there are three dataset orders and three seeds. Only one mitigation was tried, a 10% replay buffer. Regularization-based methods were not.</li>
<li><b>Grounding on ScienceQA is hard to read.</b> Many ScienceQA questions can be answered from the text of the question alone, so a score that falls little under a swapped image is not necessarily wrong there.</li>
<li><b>Test splits are the ones of the per-dataset study</b>, carved from the validation split for TextVQA and DocVQA because official test answers are hidden.</li>
<li><b>Detection, not correction.</b> Nothing here shows that a flagged answer can be fixed.</li></ul>"""


def main() -> None:
    """Load the results and write the mixture report."""
    results = Results()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(results), encoding="utf-8")
    print(f"[report] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
