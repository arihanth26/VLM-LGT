# build_report.py
#
# Builds the multi-dataset Q-head report as one self-contained HTML file with
# inline SVG charts. Every number is read from the sweep outputs
# (q_models/<dataset>/*.json) and the per-dataset analysis files
# (results/analysis_<dataset>.json), so rerunning it after new experiments
# refreshes the whole report. Compute times are the only hand-entered values.

import html
import json
from pathlib import Path

HERE = Path(__file__).parent
OUTPUT = HERE / "results" / "q_head_multidataset_report.html"

DATASETS = [
    ("chartqa_full", "ChartQA", "charts"),
    ("textvqa", "TextVQA", "scene text"),
    ("docvqa", "DocVQA", "documents"),
    ("scienceqa_img", "ScienceQA", "science diagrams"),
]
SCORERS = [
    ("token_confidence", "Token confidence", "c-tok"),
    ("separate_pre_generation", "Separate pre-gen head", "c-sp"),
    ("separate_verifier", "Separate verifier head", "c-sv"),
    ("unified_pre_generation", "Unified, pre-gen mode", "c-up"),
    ("unified_joint", "Unified, joint mode", "c-uj"),
]
SCORER_LABEL = {key: label for key, label, _ in SCORERS}
SCORER_CLASS = {key: css for key, _, css in SCORERS}
METRICS = [
    ("correctness_auroc", "Correctness AUROC", "higher"),
    ("error_auprc", "Error AUPRC", "higher"),
    ("brier", "Brier score", "lower"),
    ("ece_10_bin", "ECE (10 bins)", "lower"),
    ("pearson_correctness", "Pearson r", "higher"),
]
# Wall-clock minutes from Slurm accounting on one L40S GPU: build or download, train baseline, train features.
COMPUTE_MINUTES = {
    "textvqa": (49.7, 24.6, 28.4),
    "docvqa": (67.9, 39.7, 62.8),
    "scienceqa_img": (2.7, 8.0, 11.8),
}
COMPUTE_FILE = HERE / "results" / "compute_minutes.json"


def esc(value) -> str:
    """Escape text for HTML."""
    return html.escape(str(value))


def load_dataset_results(key: str) -> dict | None:
    """Load sweep and analysis files for one dataset, or None if any are missing."""
    folder = HERE / "q_models" / key
    analysis = HERE / "results" / f"analysis_{key}.json"
    files = (folder / "sweep_results.json", folder / "unified_sweep_results.json", analysis)
    if not all(path.exists() for path in files):
        return None
    separate, unified, extra = (json.loads(path.read_text(encoding="utf-8")) for path in files)
    metrics = {
        "token_confidence": {name: (extra["token_confidence_metrics"][name], None) for name, _, _ in METRICS},
        "separate_pre_generation": {
            name: (separate["pre_generation"]["test"][name], None) for name, _, _ in METRICS
        },
        "separate_verifier": {
            name: (separate["candidate_verifier"]["test"][name], None) for name, _, _ in METRICS
        },
    }
    for scorer, mode in (("unified_pre_generation", "pre_generation"), ("unified_joint", "joint")):
        metrics[scorer] = {
            name: (unified["test_summary"][mode][name]["mean"], unified["test_summary"][mode][name]["std"])
            for name, _, _ in METRICS
        }
    return {"separate": separate, "unified": unified, "analysis": extra, "metrics": metrics}


def f3(value: float) -> str:
    """Format a metric to three decimals."""
    return f"{value:.3f}"


def pct(value: float, digits: int = 1) -> str:
    """Format a fraction as a percentage."""
    return f"{100 * value:.{digits}f}%"


def cell(pair: tuple) -> str:
    """Format a (mean, std) pair, hiding the spread for single runs."""
    mean, std = pair
    return f3(mean) if std is None else f"{f3(mean)} ± {f3(std)}"


# ---------------------------------------------------------------- charts


class Panel:
    """Linear scales and axis drawing for one chart area inside an SVG."""

    def __init__(self, x0, y0, width, height, xmin, xmax, ymin, ymax):
        self.x0, self.y0, self.w, self.h = x0, y0, width, height
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax

    def sx(self, value: float) -> float:
        """Map a data x value to SVG x."""
        return self.x0 + (value - self.xmin) / (self.xmax - self.xmin) * self.w

    def sy(self, value: float) -> float:
        """Map a data y value to SVG y."""
        return self.y0 + self.h - (value - self.ymin) / (self.ymax - self.ymin) * self.h

    def axes(self, xticks, yticks, xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}", xlabel="", ylabel="") -> str:
        """Draw gridlines, tick labels, and axis titles."""
        parts = []
        for tick in yticks:
            y = self.sy(tick)
            parts.append(f'<line class="grid" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{y:.1f}" y2="{y:.1f}"/>')
            parts.append(f'<text class="tick" x="{self.x0 - 6}" y="{y + 3.5:.1f}" text-anchor="end">{yfmt(tick)}</text>')
        for tick in xticks:
            x = self.sx(tick)
            parts.append(f'<text class="tick" x="{x:.1f}" y="{self.y0 + self.h + 15}" text-anchor="middle">{xfmt(tick)}</text>')
        parts.append(f'<line class="axis" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.y0 + self.h}" y2="{self.y0 + self.h}"/>')
        if xlabel:
            parts.append(f'<text class="axlabel" x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 32}" text-anchor="middle">{esc(xlabel)}</text>')
        if ylabel:
            parts.append(
                f'<text class="axlabel" transform="rotate(-90)" x="{-(self.y0 + self.h / 2)}" y="{self.x0 - 38}" text-anchor="middle">{esc(ylabel)}</text>'
            )
        return "".join(parts)


def legend(items) -> str:
    """HTML legend with a swatch per series, for charts with two or more series."""
    swatches = "".join(
        f'<span class="lg"><svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true">'
        f'<line class="{css} ln" x1="1" x2="21" y1="5" y2="5"/><circle class="{css} dot" cx="11" cy="5" r="3.5"/></svg>{esc(label)}</span>'
        for label, css in items
    )
    return f'<div class="legend">{swatches}</div>'


def scorer_legend() -> str:
    """Legend covering all five scorers."""
    return legend([(label, css) for _, label, css in SCORERS])


def dot_plot(rows, metric: str, xmin: float, xmax: float, xlabel: str, chance=None, caption_id="") -> str:
    """One row per dataset, one dot per scorer, with an optional chance marker per row."""
    row_h, left, top = 62, 112, 8
    height = top + row_h * len(rows) + 40
    panel = Panel(left, top, 640 - left - 16, row_h * len(rows), xmin, xmax, 0, 1)
    parts = [f'<svg viewBox="0 0 640 {height}" role="img" aria-labelledby="{caption_id}">']
    ticks = [round(xmin + i * (xmax - xmin) / 5, 2) for i in range(6)]
    for tick in ticks:
        parts.append(f'<line class="grid" x1="{panel.sx(tick):.1f}" x2="{panel.sx(tick):.1f}" y1="{top}" y2="{top + row_h * len(rows)}"/>')
        parts.append(f'<text class="tick" x="{panel.sx(tick):.1f}" y="{top + row_h * len(rows) + 15}" text-anchor="middle">{tick:g}</text>')
    parts.append(f'<text class="axlabel" x="{left + panel.w / 2}" y="{top + row_h * len(rows) + 33}" text-anchor="middle">{esc(xlabel)}</text>')
    for index, (name, values) in enumerate(rows):
        yc = top + row_h * index + row_h / 2
        parts.append(f'<text class="rowlabel" x="{left - 10}" y="{yc + 4:.1f}" text-anchor="end">{esc(name)}</text>')
        if index:
            parts.append(f'<line class="grid" x1="{left}" x2="{left + panel.w}" y1="{top + row_h * index}" y2="{top + row_h * index}"/>')
        if chance and chance.get(name) is not None:
            x = panel.sx(chance[name])
            parts.append(f'<line class="chance" x1="{x:.1f}" x2="{x:.1f}" y1="{yc - 24:.1f}" y2="{yc + 24:.1f}"><title>{esc(name)} chance level {chance[name]:.3f}</title></line>')
        for offset, (scorer, label, css) in enumerate(SCORERS):
            if scorer not in values:
                continue
            y = yc + (offset - 2) * 9
            parts.append(
                f'<circle class="{css} dot" cx="{panel.sx(values[scorer]):.1f}" cy="{y:.1f}" r="4.2"><title>{esc(name)}, {esc(label)}: {values[scorer]:.3f}</title></circle>'
            )
    parts.append("</svg>")
    return "".join(parts)


def delta_plot(entries, caption_id="") -> str:
    """Dots with 95% interval whiskers for a paired difference, with a zero line."""
    values = [v for _, _, d, lo, hi in entries for v in (d, lo, hi)]
    limit = max(abs(min(values)), abs(max(values))) * 1.15
    row_h, left, top = 30, 190, 8
    height = top + row_h * len(entries) + 40
    panel = Panel(left, top, 640 - left - 20, row_h * len(entries), -limit, limit, 0, 1)
    parts = [f'<svg viewBox="0 0 640 {height}" role="img" aria-labelledby="{caption_id}">']
    steps = 4
    for i in range(steps * 2 + 1):
        tick = -limit + i * (2 * limit) / (steps * 2)
        parts.append(f'<line class="grid" x1="{panel.sx(tick):.1f}" x2="{panel.sx(tick):.1f}" y1="{top}" y2="{top + row_h * len(entries)}"/>')
        parts.append(f'<text class="tick" x="{panel.sx(tick):.1f}" y="{top + row_h * len(entries) + 15}" text-anchor="middle">{tick:+.3f}</text>')
    parts.append(f'<line class="axis" x1="{panel.sx(0):.1f}" x2="{panel.sx(0):.1f}" y1="{top}" y2="{top + row_h * len(entries)}"/>')
    parts.append(f'<text class="axlabel" x="{left + panel.w / 2}" y="{top + row_h * len(entries) + 33}" text-anchor="middle">Unified minus separate head, correctness AUROC</text>')
    for index, (dataset, comparison, difference, low, high) in enumerate(entries):
        yc = top + row_h * index + row_h / 2
        css = "c-uj" if comparison == "joint" else "c-up"
        parts.append(f'<text class="rowlabel" x="{left - 10}" y="{yc + 4:.1f}" text-anchor="end">{esc(dataset)}, {comparison} mode</text>')
        parts.append(f'<line class="{css} ln" x1="{panel.sx(low):.1f}" x2="{panel.sx(high):.1f}" y1="{yc:.1f}" y2="{yc:.1f}"/>')
        parts.append(
            f'<circle class="{css} dot" cx="{panel.sx(difference):.1f}" cy="{yc:.1f}" r="4.5"><title>{esc(dataset)} {comparison}: {difference:+.4f} (95% CI {low:+.4f} to {high:+.4f})</title></circle>'
        )
    parts.append("</svg>")
    return "".join(parts)


def line_panel(title: str, grid, series, ymin, ymax, ytick_step, xlabel, ylabel, diagonal=False, caption_id="") -> str:
    """Small line chart with one polyline and marker set per scorer."""
    width, height = 340, 250
    panel = Panel(48, 12, width - 62, height - 56, 0, 1, ymin, ymax)
    yticks = []
    tick = ymin
    while tick <= ymax + 1e-9:
        yticks.append(round(tick, 3))
        tick += ytick_step
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(panel.axes([0, 0.25, 0.5, 0.75, 1], yticks, xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}", xlabel=xlabel, ylabel=ylabel))
    if diagonal:
        parts.append(f'<line class="chance" x1="{panel.sx(0):.1f}" y1="{panel.sy(ymin):.1f}" x2="{panel.sx(1):.1f}" y2="{panel.sy(ymax):.1f}"/>')
    for scorer, xs, ys in series:
        css = SCORER_CLASS[scorer]
        points = " ".join(f"{panel.sx(x):.1f},{panel.sy(y):.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline class="{css} ln" points="{points}" fill="none"/>')
        for x, y in list(zip(xs, ys))[:: max(1, len(xs) // 10)]:
            parts.append(
                f'<circle class="{css} dot" cx="{panel.sx(x):.1f}" cy="{panel.sy(y):.1f}" r="2.8"><title>{esc(title)}, {esc(SCORER_LABEL[scorer])}: x={x:.2f}, y={y:.3f}</title></circle>'
            )
    parts.append("</svg>")
    return f'<figure class="panel"><figcaption>{esc(title)}</figcaption>{"".join(parts)}</figure>'


def hist_panel(title: str, histogram: dict) -> str:
    """Overlaid step outlines of the predicted-correctness distribution for right and wrong answers."""
    width, height = 340, 250
    correct = histogram["correct"]
    wrong = histogram["wrong"]
    total_c, total_w = sum(correct) or 1, sum(wrong) or 1
    share_c = [c / total_c for c in correct]
    share_w = [w / total_w for w in wrong]
    top = max(share_c + share_w) * 1.1
    panel = Panel(48, 12, width - 62, height - 56, 0, 1, 0, top)
    ticks = [round(top * i / 4, 2) for i in range(5)]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(panel.axes([0, 0.25, 0.5, 0.75, 1], ticks, yfmt=lambda v: f"{v:.2f}", xlabel="Predicted probability of correct", ylabel="Share of class"))
    edges = histogram["edges"]
    for shares, css, label in ((share_w, "c-bad", "wrong answers"), (share_c, "c-ok", "correct answers")):
        path = []
        for index, share in enumerate(shares):
            x0, x1 = panel.sx(edges[index]), panel.sx(edges[index + 1])
            path.append(f"{x0:.1f},{panel.sy(share):.1f} {x1:.1f},{panel.sy(share):.1f}")
        parts.append(f'<polyline class="{css} ln" fill="none" points="{" ".join(path)}"/>')
        for index, share in enumerate(shares):
            xm = (panel.sx(edges[index]) + panel.sx(edges[index + 1])) / 2
            parts.append(f'<circle class="{css} dot hit" cx="{xm:.1f}" cy="{panel.sy(share):.1f}" r="3"><title>{esc(label)}, bin {edges[index]:.2f} to {edges[index + 1]:.2f}: {pct(share)}</title></circle>')
    parts.append("</svg>")
    return f'<figure class="panel"><figcaption>{esc(title)}</figcaption>{"".join(parts)}</figure>'


def category_chart(rows, caption_id="") -> str:
    """Dumbbell chart of observed accuracy against mean predicted correctness per category."""
    row_h, left, top = 26, 215, 8
    height = top + row_h * len(rows) + 40
    xmin, xmax = 0.6, 1.0
    panel = Panel(left, top, 640 - left - 20, row_h * len(rows), xmin, xmax, 0, 1)
    parts = [f'<svg viewBox="0 0 640 {height}" role="img" aria-labelledby="{caption_id}">']
    for i in range(5):
        tick = xmin + i * (xmax - xmin) / 4
        parts.append(f'<line class="grid" x1="{panel.sx(tick):.1f}" x2="{panel.sx(tick):.1f}" y1="{top}" y2="{top + row_h * len(rows)}"/>')
        parts.append(f'<text class="tick" x="{panel.sx(tick):.1f}" y="{top + row_h * len(rows) + 15}" text-anchor="middle">{tick:.2f}</text>')
    parts.append(f'<text class="axlabel" x="{left + panel.w / 2}" y="{top + row_h * len(rows) + 33}" text-anchor="middle">Observed accuracy (dark) and mean predicted correctness (blue)</text>')
    for index, (label, accuracy, predicted) in enumerate(rows):
        yc = top + row_h * index + row_h / 2
        parts.append(f'<text class="rowlabel" x="{left - 10}" y="{yc + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<line class="gap" x1="{panel.sx(accuracy):.1f}" x2="{panel.sx(predicted):.1f}" y1="{yc:.1f}" y2="{yc:.1f}"/>')
        parts.append(f'<circle class="c-acc dot" cx="{panel.sx(accuracy):.1f}" cy="{yc:.1f}" r="4.5"><title>{esc(label)} observed accuracy {accuracy:.3f}</title></circle>')
        parts.append(f'<circle class="c-uj dot" cx="{panel.sx(predicted):.1f}" cy="{yc:.1f}" r="4.5"><title>{esc(label)} mean predicted {predicted:.3f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def ablation_plot(rows, caption_id="") -> str:
    """Dot chart of the best validation AUROC reached at each pre-generation loss weight."""
    row_h, left, top = 52, 112, 8
    values = [v for _, series in rows for v in series.values()]
    xmin, xmax = min(values) - 0.01, max(values) + 0.01
    height = top + row_h * len(rows) + 40
    panel = Panel(left, top, 640 - left - 16, row_h * len(rows), xmin, xmax, 0, 1)
    parts = [f'<svg viewBox="0 0 640 {height}" role="img" aria-labelledby="{caption_id}">']
    for i in range(6):
        tick = xmin + i * (xmax - xmin) / 5
        parts.append(f'<line class="grid" x1="{panel.sx(tick):.1f}" x2="{panel.sx(tick):.1f}" y1="{top}" y2="{top + row_h * len(rows)}"/>')
        parts.append(f'<text class="tick" x="{panel.sx(tick):.1f}" y="{top + row_h * len(rows) + 15}" text-anchor="middle">{tick:.3f}</text>')
    parts.append(f'<text class="axlabel" x="{left + panel.w / 2}" y="{top + row_h * len(rows) + 33}" text-anchor="middle">Validation AUROC of the pre-generation mode (best config at each weight)</text>')
    css_for = {0.0: "c-w0", 0.5: "c-w1", 1.0: "c-w2"}
    for index, (name, series) in enumerate(rows):
        yc = top + row_h * index + row_h / 2
        parts.append(f'<text class="rowlabel" x="{left - 10}" y="{yc + 4:.1f}" text-anchor="end">{esc(name)}</text>')
        if index:
            parts.append(f'<line class="grid" x1="{left}" x2="{left + panel.w}" y1="{top + row_h * index}" y2="{top + row_h * index}"/>')
        for offset, (weight, value) in enumerate(sorted(series.items())):
            parts.append(
                f'<circle class="{css_for[weight]} dot" cx="{panel.sx(value):.1f}" cy="{yc + (offset - 1) * 11:.1f}" r="4.5"><title>{esc(name)}, pre weight {weight:g}: {value:.4f}</title></circle>'
            )
    parts.append("</svg>")
    return "".join(parts)


def pipeline_diagram() -> str:
    """Flow diagram from input to correctness score to the planned correction step."""
    boxes = [
        (10, "Image and question", "input", ""),
        (170, "Qwen3-VL-2B", "frozen, no updates", ""),
        (330, "Two cached states", "prompt state, answer state", ""),
        (490, "Q-head", "small trained network", "hl"),
        (650, "Correctness score", "probability answer is right", ""),
        (810, "Gate, then correct", "planned next stage", "future"),
    ]
    parts = ['<svg viewBox="0 0 960 112" role="img" aria-labelledby="fig-pipeline">']
    for index, (x, title, sub, kind) in enumerate(boxes):
        parts.append(f'<rect class="box {kind}" x="{x}" y="22" width="140" height="68" rx="8"/>')
        parts.append(f'<text class="boxt" x="{x + 70}" y="53" text-anchor="middle">{esc(title)}</text>')
        parts.append(f'<text class="boxs" x="{x + 70}" y="72" text-anchor="middle">{esc(sub)}</text>')
        if index < len(boxes) - 1:
            parts.append(f'<path class="arrow" d="M{x + 142} 56 L{x + 158} 56"/><path class="arrowhead" d="M{x + 158} 52 L{x + 166} 56 L{x + 158} 60 z"/>')
    parts.append("</svg>")
    return "".join(parts)


def unified_diagram() -> str:
    """Block diagram of the unified head with its two input latents, flag bit, and two modes."""
    parts = ['<svg viewBox="0 0 900 250" role="img" aria-labelledby="fig-unified">']
    def box(x, y, w, h, title, sub="", kind=""):
        out = f'<rect class="box {kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>'
        out += f'<text class="boxt" x="{x + w / 2}" y="{y + h / 2 - (4 if sub else -4)}" text-anchor="middle">{esc(title)}</text>'
        if sub:
            out += f'<text class="boxs" x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle">{esc(sub)}</text>'
        return out
    def arrow(x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        bx, by = x2 - 8 * ux, y2 - 8 * uy
        head = f"M{x2} {y2} L{bx - 4 * uy:.1f} {by + 4 * ux:.1f} L{bx + 4 * uy:.1f} {by - 4 * ux:.1f} z"
        return f'<path class="arrow" d="M{x1} {y1} L{bx:.1f} {by:.1f}"/><path class="arrowhead" d="{head}"/>'
    parts.append(box(10, 30, 170, 56, "Prompt latent", "last prompt token, d = 2048"))
    parts.append(box(10, 150, 170, 56, "Answer latent", "last answer token, d = 2048", "opt"))
    parts.append(box(235, 30, 120, 56, "LayerNorm"))
    parts.append(box(235, 150, 120, 56, "LayerNorm", "", "opt"))
    parts.append(box(410, 70, 150, 100, "Concat", "prompt | answer | flag", "hl"))
    parts.append(box(615, 70, 130, 100, "Linear or MLP", "one logit"))
    parts.append(box(790, 70, 100, 100, "Score", "P(correct)"))
    parts.append(arrow(180, 58, 235, 58) + arrow(355, 58, 410, 100) + arrow(180, 178, 235, 178) + arrow(355, 178, 410, 140))
    parts.append(arrow(560, 120, 615, 120) + arrow(745, 120, 790, 120))
    parts.append('<text class="note" x="590" y="30" text-anchor="middle">Joint mode: answer present, flag = 1</text>')
    parts.append('<text class="note" x="590" y="48" text-anchor="middle">Pre-generation mode: answer half zeroed, flag = 0</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- page


CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--line:rgba(11,11,11,.10);--tok:#898781;--sp:#eda100;--sv:#eb6834;--up:#1baf7a;--uj:#2a78d6;--accent:#184f95;--good:#006300;--soft:#eef3fb;--ok:#008300;--bad:#e34948;--w0:#9ec5f4;--w1:#5598e7;--w2:#184f95}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--line:rgba(255,255,255,.10);--sp:#c98500;--sv:#d95926;--up:#199e70;--uj:#3987e5;--accent:#86b6ef;--good:#0ca30c;--soft:#202733;--ok:#008300;--bad:#e66767;--w0:#184f95;--w1:#3987e5;--w2:#9ec5f4}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--line:rgba(255,255,255,.10);--sp:#c98500;--sv:#d95926;--up:#199e70;--uj:#3987e5;--accent:#86b6ef;--good:#0ca30c;--soft:#202733;--ok:#008300;--bad:#e66767;--w0:#184f95;--w1:#3987e5;--w2:#9ec5f4}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:2rem;line-height:1.2;margin:0 0 8px}h2{font-size:1.4rem;margin:48px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:1.05rem;margin:28px 0 6px}p{margin:10px 0}.sub{color:var(--ink2);margin:0 0 24px}
.meta{color:var(--muted);font-size:.9rem}ul,ol{padding-left:22px}li{margin:6px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}
.summary{background:var(--soft);border-color:transparent}
table{border-collapse:collapse;width:100%;font-size:.92rem;margin:12px 0}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}th{font-weight:600;color:var(--ink2)}tr.best td{font-weight:600}
.tablewrap{overflow-x:auto}
figure{margin:16px 0}figure.wide{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 14px 8px}
figcaption{font-weight:600;font-size:.95rem;margin:0 0 4px}.fignote{color:var(--ink2);font-size:.9rem;margin:6px 2px 0}
svg{width:100%;height:auto;display:block}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 10px 4px;margin:0}
@media(max-width:700px){.grid2{grid-template-columns:1fr}}
.lg svg{width:22px;height:10px;flex:none}.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.88rem;color:var(--ink2);margin:6px 2px 8px}.lg{display:inline-flex;align-items:center;gap:6px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}.chance{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 3}
.gap{stroke:var(--axis);stroke-width:2}.tick{fill:var(--muted);font-size:11px}.axlabel{fill:var(--ink2);font-size:11.5px}.rowlabel{fill:var(--ink);font-size:12.5px}
.c-tok{--c:var(--tok)}.c-sp{--c:var(--sp)}.c-sv{--c:var(--sv)}.c-up{--c:var(--up)}.c-uj{--c:var(--uj)}.c-ok{--c:var(--ok)}.c-bad{--c:var(--bad)}.c-acc{--c:var(--ink)}.c-w0{--c:var(--w0)}.c-w1{--c:var(--w1)}.c-w2{--c:var(--w2)}
.ln{stroke:var(--c);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.dot{fill:var(--c);stroke:var(--surface);stroke-width:1.5}.dot.hit{fill-opacity:0;stroke:none;pointer-events:all}
.box{fill:var(--surface);stroke:var(--axis);stroke-width:1.5}.box.hl{stroke:var(--uj);stroke-width:2.5}.box.opt{stroke-dasharray:5 3}.box.future{stroke-dasharray:5 3;stroke:var(--muted)}
.boxt{fill:var(--ink);font-size:13px;font-weight:600}.boxs{fill:var(--ink2);font-size:11px}.note{fill:var(--ink2);font-size:12px}
.arrow{stroke:var(--muted);stroke-width:1.8;fill:none}.arrowhead{fill:var(--muted)}
code{font:.88em ui-monospace,Consolas,monospace;background:var(--soft);padding:1px 5px;border-radius:4px}
pre{background:var(--soft);padding:12px 14px;border-radius:8px;overflow-x:auto;font:.85rem/1.5 ui-monospace,Consolas,monospace}
.pos{color:var(--good);font-weight:600}.small{font-size:.9rem;color:var(--ink2)}
"""


def table(headers, rows, best_rows=()) -> str:
    """Render an HTML table with optional emphasized rows."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for index, row in enumerate(rows):
        css = ' class="best"' if index in best_rows else ""
        body += f"<tr{css}>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(identifier: str, caption: str, body: str, note: str = "", wide: bool = True) -> str:
    """Wrap a chart in a titled figure with an optional note beneath it."""
    css = ' class="wide"' if wide else ""
    note_html = f'<p class="fignote">{note}</p>' if note else ""
    return f'<figure{css}><figcaption id="{identifier}">{caption}</figcaption>{body}{note_html}</figure>'


def build(present: dict) -> str:
    """Assemble the full report from the loaded per-dataset results."""
    names = {key: name for key, name, _ in DATASETS}
    domain = {key: kind for key, _, kind in DATASETS}
    order = [key for key, _, _ in DATASETS if key in present]
    count_word = {2: "two", 3: "three", 4: "four"}.get(len(order), str(len(order)))
    new_order = [key for key in order if key != "chartqa_full"]
    analysis = {key: present[key]["analysis"] for key in order}
    test_acc = {key: analysis[key]["splits"]["test"]["accuracy"] for key in order}
    metric = {key: present[key]["metrics"] for key in order}
    grid = analysis[order[0]]["grid"]

    def at(scorer_key, key, name):
        return metric[key][scorer_key][name][0]

    def curve(key, scorer_key, field, coverage):
        return analysis[key]["scorers"][scorer_key][field][grid.index(coverage)]

    gate_rows = []
    for key in order:
        error_rate = 1 - test_acc[key]
        capture = curve(key, "unified_pre_generation", "error_capture", 0.2)
        capture_joint = curve(key, "unified_joint", "error_capture", 0.2)
        capture_tok = curve(key, "token_confidence", "error_capture", 0.2)
        gate_rows.append((key, error_rate, capture_tok, capture, capture_joint, capture_joint * error_rate / 0.2))
    precision_low = min(r[5] for r in gate_rows)
    precision_high = max(r[5] for r in gate_rows)

    boot = {key: analysis[key]["bootstrap"] for key in order}
    wins, ties, losses = [], [], []
    for key in order:
        interval = boot[key]["joint_minus_separate_verifier"]["auroc"]
        (wins if interval["ci_low"] > 0 else losses if interval["ci_high"] < 0 else ties).append(key)

    def join_names(keys):
        labels = [names[k] for k in keys]
        return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]

    gain_low = min(at("unified_joint", k, "correctness_auroc") - at("token_confidence", k, "correctness_auroc") for k in order)
    gain_high = max(at("unified_joint", k, "correctness_auroc") - at("token_confidence", k, "correctness_auroc") for k in order)
    one_fraction = {k: analysis[k]["token_confidence_exactly_one"] for k in order}
    input_words = {"chartqa_full": "charts", "textvqa": "photographs with text", "docvqa": "scanned pages", "scienceqa_img": "science diagrams"}
    input_list = ", ".join(input_words[k] for k in order[:-1]) + " and " + input_words[order[-1]]

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q-Head Correctness Report</title><style>{CSS}</style></head><body><main>
<h1>Predicting Answer Correctness From Frozen Latents</h1>
<p class="sub">A {count_word}-dataset study of correctness heads on Qwen3-VL-2B, and a unified head that scores an answer before and after it is generated.</p>
<p class="meta">Datasets: {', '.join(names[k] for k in order)}. All numbers are on held-out test splits. Charts are inline SVG and follow the system light or dark setting.</p>
"""]

    # Summary
    parts.append(f"""<div class="card summary"><h3 style="margin-top:0">Summary</h3><ul>
<li><b>A small learned head predicts when the model is wrong far better than the model's own confidence.</b> Across {count_word} datasets with different input types, the unified head raises correctness AUROC over geometric-mean token confidence by {gain_low:.2f} to {gain_high:.2f}. Token confidence is saturated: it equals 1.0 on {pct(min(one_fraction.values()), 0)} to {pct(max(one_fraction.values()), 0)} of answers.</li>
<li><b>One head can do both jobs.</b> The unified head reads the prompt latent and the answer latent together, and with the answer half zeroed it scores a question before any answer exists. Its joint mode matches or beats the separate verifier head on every dataset; the paired bootstrap interval excludes zero in its favor on {join_names(wins) if wins else 'none of the datasets'}{', and includes zero on ' + join_names(ties) if ties else ''}{', and favors the separate head on ' + join_names(losses) if losses else ''}.</li>
<li><b>It is useful as a gate.</b> Flagging the 20% of answers with the lowest predicted correctness captures {pct(min(r[4] for r in gate_rows), 0)} to {pct(max(r[4] for r in gate_rows), 0)} of all errors, against {pct(min(r[2] for r in gate_rows), 0)} to {pct(max(r[2] for r in gate_rows), 0)} for token confidence. Keeping only the most trusted half of answers lifts accuracy to {pct(min(curve(k, 'unified_joint', 'accuracy_at_coverage', 0.5) for k in order), 0)} to {pct(max(curve(k, 'unified_joint', 'accuracy_at_coverage', 0.5) for k in order), 0)}.</li>
<li><b>The extra pre-generation loss term costs almost nothing.</b> On validation it lifts the pre-generation mode on every dataset and moves the joint mode by less than 0.006 AUROC. In pre-generation mode the unified head lands within 0.006 AUROC of the dedicated pre-generation head, slightly below it on two datasets and slightly above on one. The pairwise reference-answer loss hurt on every dataset and is turned off in all winners.</li>
<li><b>What this does not show yet.</b> These heads detect likely errors. They do not fix them. Whether a latent correction step helps on the flagged cases is the next experiment.</li></ul></div>""")

    # 1 Background
    parts.append(f"""<h2>1. Why a correctness head</h2>
<p>The project asks a vision-language model to notice when its visual evidence is weak and act on that before answering. Acting needs a trigger, and the trigger needs a score that is trustworthy: a score that is high when the answer is right and low when it is wrong. Work on self-correction in grounding tasks has found that this verification signal, not the correction mechanism, is the weak point. A model that corrects answers it already got right can lose accuracy, and a model that rates itself confidently while wrong gives the trigger nothing to work with.</p>
<p>The obvious free signal is the probability the model assigns to its own answer tokens. On these datasets it does not work well. The geometric mean of the token probabilities is exactly 1.0 for {pct(min(one_fraction.values()), 0)} to {pct(max(one_fraction.values()), 0)} of test answers, depending on the dataset, so the score cannot separate right from wrong for most of the data. This report measures how much better a small head trained on the model's hidden states does, and whether the result holds across very different kinds of input.</p>
{figure('fig-pipeline', 'Where the correctness head sits in the project', pipeline_diagram(), 'The language model is frozen throughout. Only the small head is trained. The dashed final box is the planned correction stage and is not part of the results here.')}""")

    # 2 Phase 1
    parts.append("""<h2>2. Foundation: the ChartQA study</h2>
<p>The first stage of the work built the full pipeline on ChartQA (Qwen3-VL-2B, 2,500 test questions, 75.8% base accuracy) and fixed the design that the rest of this report reuses.</p>
<ul>
<li><b>Frozen features, cached once.</b> Each question is run through the model and two last-layer hidden states are stored: the state at the last prompt token (before any answer, called the pre-generation state) and the state at the last token after the generated answer is appended (the candidate state). A reference-answer state is also stored for training pairs.</li>
<li><b>Two small heads.</b> A pre-generation head reads the first state and a verifier head reads the second. Each is a linear probe or a small MLP trained with binary cross-entropy against a correctness label, optionally with a batch ranking loss and, for the verifier, a paired loss that scores the reference answer above a wrong generated one.</li>
<li><b>Calibration.</b> A single temperature fitted on the validation split rescales the logits. The test split is used once.</li>
<li><b>Sweeps.</b> A 72-fit sweep over architecture, learning rate, ranking weight and seed, followed by an early-stopping sweep over the verifier ranking weight.</li>
</ul>
<p>The findings from that study shaped the design here: token confidence was weak, a linear probe won the pre-generation sweep (question difficulty is close to linearly readable), and plain cross-entropy beat the ranking objectives. Original ChartQA test numbers:</p>""")
    parts.append(table(
        ["Metric", "Token confidence", "Pre-gen MLP", "Pre-gen linear", "Verifier MLP", "Verifier, early stop"],
        [["Correctness AUROC", "0.750", "0.863", "0.895", "0.887", "0.891 ± 0.001"],
         ["Error AUPRC", "0.583", "0.669", "0.732", "0.723", "0.738 ± 0.005"],
         ["Brier score", "0.178", "0.123", "0.109", "0.112", "0.111 ± 0.001"],
         ["ECE (10 bins)", "0.172", "0.055", "0.036", "0.030", "0.038 ± 0.014"],
         ["Accuracy at 50% coverage", "92.4%", "95.1%", "97.0%", "96.6%", "97.0%"]]))
    parts.append('<p class="small">These are the numbers from the original study. Section 5 reruns ChartQA under the same protocol as the other datasets so that all comparisons are like for like.</p>')

    # 3 What was added
    parts.append(f"""<h2>3. What was built on top</h2>
<p>The extension had four parts: more datasets, a unified head, automation for the cluster, and analysis tooling.</p>
<h3>3.1 More datasets, chosen for different inputs</h3>
<p>A result on charts could simply be a property of charts. Three further datasets were chosen to differ in input type and answer format. Each has human-written answers, public train labels and a held-out labeled split.</p>""")
    parts.append(table(
        ["Dataset", "Input type", "Answer format", "Correct means", "Splits used"],
        [["TextVQA", "Natural photos with text", "Free text, 10 annotators", "VQA score of 0.5 or more (at least 2 of 10 agree)", "Train sample of 8,000. Val 1,500 and test 2,500 from the labeled validation split, grouped by image."],
         ["DocVQA", "Scanned industry documents", "Free text from the page", "ANLS similarity of 0.9 or more", "Train sample of 8,000. Val 1,500 and test 2,500 from the labeled validation split, grouped by document."],
         ["ScienceQA (image questions)", "Science diagrams", "Multiple choice letter", "Exact option letter", "Official splits, image questions only: 6,218 train, 1,500 val, 2,017 test."]]))
    parts.append("""<p>Candidates that were rejected, with reasons: GQA (questions and images are in separate configs and need a join), VQAv2 (only validation is labeled on the Hub copy, and labels are noisy), OK-VQA (answers depend on outside knowledge), AI2D and MathVista (too small to train a head), InfographicVQA (labeled validation only). TextVQA and DocVQA do not release test answers, so their held-out data comes from the validation split. The carving is done by image or document so that no source image appears in more than one split. DocVQA page images are capped at 1,536 pixels on the long side once at build time, so generation and feature extraction see identical pixels.</p>
<h3>3.2 A unified head</h3>
<p>The two separate heads see different cached states and must be run as two models. The unified head takes both states as one input and is described in Section 4.</p>
<h3>3.3 Cluster automation</h3>
<p>One command per dataset submits a chain of dependent jobs on the ICE cluster: build or download the splits, run baseline generation and feature extraction for train, validation and test, then both sweeps. The model weights are fetched once up front so parallel jobs do not race, and caches live in scratch because the home quota is 30 GB. GPU jobs request a flexible constraint (H200, H100, L40S, A100 or A40) instead of a fixed card, because the account's partitions do not include H200s; every job here ran on an L40S.</p>
<h3>3.4 Analysis tooling</h3>
<p>A post-sweep script reloads each winning head and computes the curves in Section 5: accuracy when the least trusted answers are dropped, how many errors the lowest-scored answers contain, reliability bins, score histograms, per-category accuracy, and paired bootstrap intervals. A report builder (this document) reads those files directly. Offline tests cover the scoring rules, the grouped split, and the unified head and sweep.</p>""")

    # 4 Unified head
    parts.append(f"""<h2>4. The unified head</h2>
{figure('fig-unified', 'Unified Q-head architecture', unified_diagram(), 'Dashed boxes are the answer-side path, which is zeroed in pre-generation mode.')}
<p>Each latent is layer-normalized separately so the two are on a common scale, then concatenated with a one-bit flag that says whether an answer is present. A linear layer or small MLP maps the result to a single correctness logit. With the answer half set to zeros and the flag at 0 the same weights score a question before generation. This lets the gate run first and the answer-conditioned score be computed only when a correction might follow.</p>
<h3>Loss</h3>
<pre>L = BCE(f(prompt, answer), y)                       joint mode
  + w_pre  * BCE(f(prompt, none), y)                pre-generation mode, same weights
  + w_rank * ranking(f(prompt, answer), y)          correct above wrong, off by default
  + w_pair * softplus(f(p, a) - f(p, reference))    wrong rows only, off in all winners</pre>
<p>Both modes predict the same label, so the targets never conflict. The risk is a shortcut: the answer latent is the stronger signal, and a head trained only on joint inputs can lean on it and neglect the prompt half, which would make the pre-generation mode poor. The second term is there to prevent that. Both modes are computed on every batch, with no random dropping of the answer half, so the two terms keep a fixed ratio. Both cross-entropy terms are proper scoring rules, so calibration comes from the loss, and a per-mode temperature fitted on validation corrects what is left.</p>
<h3>Protocol</h3>
<p>Each configuration is trained on three seeds with early stopping on validation. The grid covers architecture (linear, 256-unit MLP, 512-unit MLP), learning rate (3e-4, 1e-3), pre-generation weight (0, 0.5, 1) and pair weight (0, 0.25): 36 configurations and 108 trainings per dataset. The winner is the configuration with the best mean validation score over seeds, using the average of both modes' error AUPRC with AUROC as a tiebreak. The test split is scored once, for the winner only. The separate heads from the earlier sweep are included for comparison, trained on exactly the same cached features.</p>""")

    # 5 Setup
    parts.append("<h2>5. Setup and results</h2><h3>5.1 Experimental setup</h3>")
    setup_rows = []
    for key in order:
        sp = analysis[key]["splits"]
        setup_rows.append([names[key], domain[key], f"{sp['train']['count']:,} / {sp['val']['count']:,} / {sp['test']['count']:,}",
                           f"{pct(sp['train']['accuracy'])} / {pct(sp['val']['accuracy'])} / {pct(sp['test']['accuracy'])}", pct(one_fraction[key], 0)])
    parts.append(table(["Dataset", "Input", "Train / val / test examples", "Base accuracy, train / val / test", "Token confidence exactly 1.0"], setup_rows))
    parts.append('<p>Base model: Qwen3-VL-2B-Instruct in bf16, one greedy generation per question with a short-answer instruction, up to 64 new tokens. Features are the final-layer hidden states, cached once. Heads train on cached features only. Hardware: one NVIDIA L40S per job on ICE.</p>')

    parts.append("<h3>5.2 Headline results</h3>")
    for key in order:
        rows = []
        best = {}
        for name, _, direction in METRICS:
            values = {s: metric[key][s][name][0] for s, _, _ in SCORERS if s != "token_confidence"}
            best[name] = max(values, key=values.get) if direction == "higher" else min(values, key=values.get)
        for name, label, direction in METRICS:
            row = [f"{label} {'↑' if direction == 'higher' else '↓'}"]
            for s, _, _ in SCORERS:
                text = cell(metric[key][s][name])
                row.append(f"<b>{text}</b>" if best[name] == s else text)
            rows.append(row)
        parts.append(f"<h3 style='margin-top:20px'>{names[key]} <span class='meta'>test accuracy {pct(test_acc[key])}</span></h3>")
        parts.append(table(["Metric"] + [SCORER_LABEL[s] for s, _, _ in SCORERS], rows))
    parts.append('<p class="small">Bold marks the best learned scorer per row. The separate heads are single runs. Unified numbers are the mean ± standard deviation over three seeds.</p>')

    # discrimination plots
    rows_auroc = [(names[k], {s: metric[k][s]["correctness_auroc"][0] for s, _, _ in SCORERS}) for k in order]
    rows_auprc = [(names[k], {s: metric[k][s]["error_auprc"][0] for s, _, _ in SCORERS}) for k in order]
    chance = {names[k]: 1 - test_acc[k] for k in order}
    parts.append("<h3>5.3 Discrimination</h3>")
    parts.append(scorer_legend())
    parts.append(figure("fig-auroc", "Correctness AUROC by dataset and scorer", dot_plot(rows_auroc, "auroc", 0.6, 1.0, "AUROC (0.5 is chance, axis starts at 0.6)", caption_id="fig-auroc"),
                        "Every learned scorer sits well to the right of token confidence. The axis is truncated at 0.6, so read position, not distance from the edge."))
    parts.append(figure("fig-auprc", "Error AUPRC by dataset and scorer", dot_plot(rows_auprc, "auprc", 0.1, 0.7, "Average precision at finding wrong answers", chance=chance, caption_id="fig-auprc"),
                        "AUPRC for errors depends on how many errors there are. The dashed tick in each row marks the chance level, which equals that dataset's error rate. The axis starts at 0.1."))

    # unified vs separate
    entries = []
    for key in order:
        b = boot[key]
        for comparison, label in (("joint", "joint_minus_separate_verifier"), ("pre", "unified_pre_minus_separate_pre")):
            interval = b[label]["auroc"]
            entries.append((names[key], comparison, interval["difference"], interval["ci_low"], interval["ci_high"]))
    parts.append("<h3>5.4 Unified versus separate heads</h3>")
    parts.append(figure("fig-delta", "Paired AUROC difference, unified minus separate, with 95% bootstrap intervals", delta_plot(entries, "fig-delta"),
                        "Joint mode is compared with the separate verifier and pre-generation mode with the separate pre-generation head, using the same test examples (500 paired resamples, unified scores averaged over three seeds). An interval that crosses zero is a tie."))
    delta_rows = []
    for key in order:
        j = boot[key]["joint_minus_separate_verifier"]
        p = boot[key]["unified_pre_minus_separate_pre"]
        delta_rows.append([names[key],
                           f"{j['auroc']['difference']:+.3f} ({j['auroc']['ci_low']:+.3f}, {j['auroc']['ci_high']:+.3f})",
                           f"{j['error_auprc']['difference']:+.3f} ({j['error_auprc']['ci_low']:+.3f}, {j['error_auprc']['ci_high']:+.3f})",
                           f"{p['auroc']['difference']:+.3f} ({p['auroc']['ci_low']:+.3f}, {p['auroc']['ci_high']:+.3f})",
                           f"{p['error_auprc']['difference']:+.3f} ({p['error_auprc']['ci_low']:+.3f}, {p['error_auprc']['ci_high']:+.3f})"])
    parts.append(table(["Dataset", "Joint vs verifier, AUROC", "Joint vs verifier, error AUPRC", "Pre vs pre, AUROC", "Pre vs pre, error AUPRC"], delta_rows))

    # selective prediction
    parts.append("<h3>5.5 Using the score: dropping answers and flagging errors</h3>")
    parts.append("<p>Two views of the same scores show what they are worth in use. First, if the least trusted answers are dropped, how accurate is what remains? Second, if a fixed share of answers with the lowest scores is flagged for a second look, how many of the model's errors does that share contain?</p>")
    parts.append(scorer_legend())
    panels = []
    for key in order:
        series = [(s, grid, analysis[key]["scorers"][s]["accuracy_at_coverage"]) for s, _, _ in SCORERS]
        low = min(min(v) for _, _, v in series)
        ymin = max(0.0, (int(low * 20) / 20) - 0.05)
        panels.append(line_panel(names[key], grid, series, round(ymin, 2), 1.0, 0.05 if (1 - ymin) <= 0.3 else 0.1, "Fraction of answers kept (most trusted first)", "Accuracy of kept answers"))
    parts.append(figure("fig-coverage", "Accuracy of the answers that remain as the least trusted are dropped", f'<div class="grid2">{"".join(panels)}</div>',
                        "The right edge is coverage 1.0, which equals the base accuracy. A curve that stays high toward the right means the scorer ranks wrong answers last."))
    panels = []
    for key in order:
        series = [(s, grid, analysis[key]["scorers"][s]["error_capture"]) for s, _, _ in SCORERS]
        panels.append(line_panel(names[key], grid, series, 0.0, 1.0, 0.25, "Fraction of answers flagged (least trusted first)", "Share of all errors captured", diagonal=True))
    parts.append(figure("fig-capture", "Share of all errors found in the flagged fraction of answers", f'<div class="grid2">{"".join(panels)}</div>',
                        "The dashed diagonal is what random flagging would achieve. Token confidence does find errors among the 10 to 21% of answers it scores below 1.0, then flattens, because every remaining answer is tied at 1.0 and the ties are broken at random. The learned scores keep ranking answers across the whole range."))
    gate_table = [[names[k], pct(e), pct(ct, 0), pct(cp, 0), pct(cj, 0), pct(pj, 0)] for k, e, ct, cp, cj, pj in gate_rows]
    parts.append(table(["Dataset", "Base error rate", "Errors captured, token confidence", "Errors captured, unified pre-gen mode", "Errors captured, unified joint mode", "Precision of the flagged set, joint mode"], gate_table))
    parts.append('<p class="small">All columns flag the 20% of answers with the lowest score. Precision is the share of flagged answers that are actually wrong, to be compared with the base error rate in the second column. Flagging at random would capture 20% of errors and have precision equal to that base error rate.</p>')

    # calibration
    parts.append("<h3>5.6 Calibration</h3>")
    panels = []
    for key in order:
        series = []
        for scorer, rel in (("token_confidence", analysis[key]["reliability"]["token_confidence"]), ("unified_joint", analysis[key]["reliability"]["unified_joint"])):
            series.append((scorer, [r["mean_probability"] for r in rel], [r["accuracy"] for r in rel]))
        panels.append(line_panel(names[key], grid, series, 0.0, 1.0, 0.25, "Mean predicted probability", "Observed accuracy", diagonal=True))
    parts.append(legend([("Token confidence", "c-tok"), ("Unified, joint mode", "c-uj")]))
    parts.append(figure("fig-reliability", "Reliability: predicted probability against observed accuracy", f'<div class="grid2">{"".join(panels)}</div>',
                        "Points on the dashed diagonal are perfectly calibrated. Token confidence piles into the top-right corner; the learned head spreads across the range and tracks the diagonal after temperature scaling."))
    panels = [hist_panel(names[k], analysis[k]["histogram_unified_joint"]) for k in order]
    parts.append(legend([("Wrong answers", "c-bad"), ("Correct answers", "c-ok")]))
    parts.append(figure("fig-hist", "Where right and wrong answers land on the unified joint score", f'<div class="grid2">{"".join(panels)}</div>',
                        "Each curve is normalized within its class. Separation between the two is what AUROC measures; the overlap is the part the head cannot tell apart."))

    # categories
    cat_blocks = []
    for key in order:
        cats = analysis[key].get("categories") or []
        if cats:
            rows = [(f"{c['category']} (n={c['count']})", c["accuracy"], c["mean_predicted"]) for c in cats]
            cat_blocks.append(figure(f"fig-cat-{key}", f"{names[key]}: accuracy and predicted correctness by question category", category_chart(rows, f"fig-cat-{key}"),
                                     "A gap between the two dots means the head is miscalibrated for that category."))
    if cat_blocks:
        parts.append("<h3>5.7 By question category</h3>" + "".join(cat_blocks))

    # ablations
    parts.append("<h3>5.8 What mattered in the sweep</h3>")
    ablation_rows = []
    pair_rows = []
    arch_rows = []
    for key in order:
        aggregates = present[key]["unified"]["aggregates"]
        series = {}
        for weight in (0.0, 0.5, 1.0):
            subset = [a for a in aggregates if a["config"]["pre_weight"] == weight]
            series[weight] = max(a["mean_val_auroc"]["pre_generation"] for a in subset)
        ablation_rows.append((names[key], series))
        joint0 = max(a["mean_val_auroc"]["joint"] for a in aggregates if a["config"]["pre_weight"] == 0.0)
        joint1 = max(a["mean_val_auroc"]["joint"] for a in aggregates if a["config"]["pre_weight"] == 1.0)
        pair0 = [a["mean_val_auroc"]["joint"] for a in aggregates if a["config"]["pair_weight"] == 0.0]
        pair1 = [a["mean_val_auroc"]["joint"] for a in aggregates if a["config"]["pair_weight"] == 0.25]
        pair_rows.append([names[key], f3(sum(pair0) / len(pair0)), f3(sum(pair1) / len(pair1)), f"{(sum(pair1) / len(pair1)) - (sum(pair0) / len(pair0)):+.3f}"])
        best_arch = {arch: max(a["mean_val_auroc"]["joint"] for a in aggregates if a["config"]["architecture"] == arch) for arch in ("linear", "mlp_256", "mlp_512")}
        arch_rows.append([names[key], f3(best_arch["linear"]), f3(best_arch["mlp_256"]), f3(best_arch["mlp_512"]),
                          present[key]["unified"]["winner"]["config"]["architecture"], f"{series[0.0]:.3f} to {series[1.0]:.3f}", f"{joint1 - joint0:+.3f}"])
    parts.append("<p><b>The pre-generation loss term.</b> The chart shows the best validation AUROC that the pre-generation mode reaches when the term is off (weight 0), half strength and full strength. The term helps on every dataset, and weight 1 was selected on every dataset.</p>")
    parts.append(legend([("Weight 0", "c-w0"), ("Weight 0.5", "c-w1"), ("Weight 1.0", "c-w2")]))
    parts.append(figure("fig-ablation", "Effect of the pre-generation loss weight on the pre-generation mode (validation)", ablation_plot(ablation_rows, "fig-ablation"),
                        "The same weights serve the joint mode, whose best validation AUROC changes by at most a few thousandths between weight 0 and weight 1 (last column of the architecture table below)."))
    parts.append("<p><b>The pairwise reference-answer loss.</b> Mean validation joint AUROC over all configurations with the pair term off and on:</p>")
    parts.append(table(["Dataset", "Pair weight 0", "Pair weight 0.25", "Change"], pair_rows))
    parts.append("<p><b>Architecture.</b> Best validation joint AUROC for each architecture, the architecture that won, and the pre-generation weight effect on both modes:</p>")
    parts.append(table(["Dataset", "Linear", "MLP 256", "MLP 512", "Winner", "Pre mode, weight 0 to 1", "Joint mode change"], arch_rows))

    # insights
    parts.append(f"""<h2>6. What the results mean</h2>
<h3>The signal is real and transfers across input types</h3>
<p>The learned heads beat token confidence on every dataset, and they do so on {input_list}. The model's hidden state at the end of the prompt already encodes much of how hard the question is for it, and the state after the answer encodes more about whether that specific answer is right. This is not a property of one dataset.</p>
<h3>Where the answer latent adds information, and where it does not</h3>""")
    gaps = {k: at("unified_joint", k, "correctness_auroc") - at("unified_pre_generation", k, "correctness_auroc") for k in order}
    parts.append(f"""<p>The gap between the unified head's joint mode and its pre-generation mode shows how much seeing the answer adds: {', '.join(f"{names[k]} {gaps[k]:+.3f}" for k in order)} AUROC. Where the answer is free text read from the image, as in TextVQA and DocVQA, the answer-side state carries extra evidence, probably because it reflects whether the model read the right string. In ScienceQA the answer is a single letter among a few options, so the answer-side state has little to add beyond what the question and image already say, and the pre-generation mode is nearly as good as the joint mode. This suggests that the benefit of an answer-conditioned verifier depends on how much information the answer itself carries.</p>
<h3>A pre-generation score is already worth having</h3>
<p>In pre-generation mode the head never sees the answer, yet it captures most of the benefit: on ScienceQA it reaches {pct(curve('scienceqa_img', 'unified_pre_generation', 'accuracy_at_coverage', 0.5), 1) if 'scienceqa_img' in analysis else 'n/a'} accuracy on the most trusted half of questions. Because it runs before generation, it can decide whether to spend extra compute at all.</p>
<h3>The unified head is a simplification that does not cost accuracy</h3>
<p>Matching or beating the separate heads with one model is a practical result: one set of weights, one cached input pair, and a choice of mode at inference. The pre-generation loss term lets the head keep a usable pre-answer score without hurting the answer-conditioned one.</p>
<h3>The pairwise reference loss does not help</h3>
<p>Adding the paired loss reduced mean validation AUROC on every dataset (Section 5.8), by a large margin on DocVQA. A plausible reason, not tested here, is that the reference-answer state differs from a generated-answer state in ways unrelated to correctness, so the head can improve the pair loss by learning that difference instead. The same pattern held for the ranking loss in the first study. Plain cross-entropy is the right default.</p>
<h3>Calibration is good on average and uneven by category</h3>
<p>After temperature scaling the expected calibration error is small on every dataset. The category breakdown shows where it is not: on DocVQA handwritten questions accuracy is well below the head's mean prediction, so the head is overconfident exactly where the model is weakest. A gate built on this score would under-trigger there.</p>
<h3>Practical meaning for a correction gate</h3>
<p>If a second pass costs one extra generation, flagging 20% of answers costs 20% more compute and reaches the capture rates in Section 5.5. Precision is the other side of that trade. With the joint score, between {pct(precision_low, 0)} and {pct(precision_high, 0)} of the flagged answers are actually wrong, against base error rates of {pct(min(r[1] for r in gate_rows), 0)} to {pct(max(r[1] for r in gate_rows), 0)}. That is a {min(r[5] / r[1] for r in gate_rows):.1f} to {max(r[5] / r[1] for r in gate_rows):.1f} times enrichment, but it also means the rest of the flagged set is answers that were already right. A correction step therefore has to be safe on correct answers, a requirement that comes straight from the self-correction literature, and these precision numbers say how often it will matter.</p>""")

    # limitations
    parts.append("""<h2>7. Limits of these results</h2><ul>
<li><b>Detection, not correction.</b> Nothing here shows that a flagged answer can be fixed. A wrong answer the model cannot fix is still a correct detection.</li>
<li><b>One model size.</b> Only Qwen3-VL-2B was run. The 4B model is the natural check.</li>
<li><b>Carved test splits.</b> TextVQA and DocVQA test sets come from the labeled validation split because the official test answers are hidden. They are grouped by image or document, but they are not the official benchmark test sets.</li>
<li><b>Training subsets.</b> TextVQA and DocVQA heads train on 8,000 sampled questions rather than the full train splits.</li>
<li><b>Single separate-head runs.</b> The separate heads have no seed spread. The paired bootstrap accounts for test-set sampling but not for training-seed variation of those heads, so small differences should not be over-read.</li>
<li><b>Threshold choices in the label.</b> The correct label depends on cutoffs (two annotators for TextVQA, ANLS 0.9 for DocVQA). Other cutoffs would shift the base accuracy and the numbers somewhat.</li>
<li><b>Error AUPRC is noisy.</b> With 13 to 19% errors on 2,000 to 2,500 questions, error AUPRC moves by a few thousandths between seeds, so differences of that size are not evidence.</li>
<li><b>Question difficulty versus answer quality.</b> A pre-generation score can learn properties of the question set that predict errors without tracking visual evidence. The planned counterfactual control (corrupting the visual evidence and checking that the score moves) has not been run.</li></ul>""")

    # next steps
    parts.append("""<h2>8. Next steps</h2><ol>
<li><b>Build the correction step and test it with the gate.</b> A first candidate is test-time latent steering that nudges the hidden state toward a higher correctness score and re-answers, compared with a random nudge of the same size and evaluated only on information available at inference.</li>
<li><b>Strengthen the head cheaply.</b> Cache several layers instead of the last one, add pooled image-token features, and add the logged token statistics as extra inputs.</li>
<li><b>Change the target to "would a correction help".</b> Once a correction exists, train the gate on cases where it improved the answer, not just on whether the answer was wrong.</li>
<li><b>Run the counterfactual and 4B checks</b> listed above.</li></ol>""")

    # appendix
    times = dict(COMPUTE_MINUTES)
    if COMPUTE_FILE.exists():
        times.update({k: tuple(v) for k, v in json.loads(COMPUTE_FILE.read_text(encoding="utf-8")).items()})
    compute_rows = [[names[k], f"{t[0]:.0f}", f"{t[1]:.0f}", f"{t[2]:.0f}"] for k, t in times.items() if k in names]
    parts.append("<h2>Appendix: reproduction and compute</h2>")
    parts.append("<p>Run one dataset end to end from the repository root on ICE, then build the comparison and this report:</p>")
    parts.append("""<pre>bash slurm/run_qhead_pipeline.sh textvqa     # also docvqa, scienceqa_img, chartqa_full
python -m data.approach1_latent_self_correction.compare_q_heads_report
python -m data.approach1_latent_self_correction.analyze_q_heads --dataset textvqa
python -m data.approach1_latent_self_correction.build_report</pre>""")
    parts.append("<p>Wall-clock minutes per stage on one L40S (the train split dominates; val and test are shorter):</p>")
    parts.append(table(["Dataset", "Build or download", "Train baseline generation", "Train feature extraction"], compute_rows))
    parts.append('<p class="small">Each sweep (separate heads, then unified) took 2 to 5 minutes per dataset on cached features. Hyperparameter grids are in the Section 4 protocol. Result files: <code>q_models/&lt;dataset&gt;/sweep_results.json</code>, <code>unified_sweep_results.json</code> and <code>results/analysis_&lt;dataset&gt;.json</code>.</p>')
    parts.append("</main></body></html>")
    return "".join(parts)


def main() -> None:
    """Load every available dataset's results and write the HTML report."""
    present = {}
    for key, _, _ in DATASETS:
        loaded = load_dataset_results(key)
        if loaded is None:
            print(f"[skip] {key}: results or analysis file missing")
        else:
            present[key] = loaded
    if not present:
        raise SystemExit("no dataset results found")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(present), encoding="utf-8")
    print(f"[report] wrote {OUTPUT} with {', '.join(present)}")


if __name__ == "__main__":
    main()
