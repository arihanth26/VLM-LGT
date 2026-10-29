# build_approach1_report.py
#
# Builds the complete Approach 1 report as one self-contained HTML file, written for a mid-point
# report and presentation. It walks the whole project in order: the answer-token probability
# baseline, the original ChartQA study, the three added datasets, the unified head, head sizes,
# the pooled mixture, the old and new loss functions (swap and noise corruption), sequential
# training and forgetting, the trained heads, and the prompted correction experiments. Every
# number is read from the result files, nothing is typed in except the original ChartQA study
# table, which exists only as a report. Math is rendered by KaTeX, which is vendored under
# assets/katex and embedded in the page, so the file works offline and prints to PDF.
# Usage: python -m experiments.approach1_latent_self_correction.build_approach1_report [--pdf]

import argparse
import base64
import html
import json
import math
import re
import subprocess
from pathlib import Path

from experiments.approach1_latent_self_correction import build_corruption_report as corr
from experiments.approach1_latent_self_correction.scoring import normalize_vqa_answer as norm
from experiments.approach1_latent_self_correction.build_mixture_report import (
    NAMES,
    Results,
    cell_change,
    heatmap,
    pair_plot,
)
from experiments.approach1_latent_self_correction.build_report import (
    CSS,
    DATASET_INFO,
    ORIGINAL_TEST,
    SCORER_LABEL,
    SCORERS as BASE_SCORERS,
    dot_plot,
    esc,
    figure,
    legend,
    line_panel,
    pipeline_diagram,
    table,
    unified_diagram,
)

SCORER_LABEL["token_confidence"] = "Answer-token probability"
SCORERS = [(key, SCORER_LABEL[key], css) for key, _, css in BASE_SCORERS]
HERE = Path(__file__).parent
REPO = HERE.parents[1]
SWEEP = REPO / "results" / "approach1" / "04_sweep_data"
RESULTS = HERE / "results"
ASSETS = HERE / "assets" / "katex"
OUTPUT = RESULTS / "approach1_complete_report.html"
RULE = {"chartqa_full": "Relaxed accuracy (5% for numbers)", "textvqa": "VQA soft accuracy at least 0.5", "docvqa": "ANLS at least 0.9", "scienceqa_img": "Option letter matches"}
DATASETS = [("chartqa_full", "ChartQA"), ("textvqa", "TextVQA"), ("docvqa", "DocVQA"), ("scienceqa_img", "ScienceQA")]
FEATURE = 2048
KINDS = ("pre", "verifier", "unified")
KIND_NAME = {"pre": "Separate pre-generation head", "verifier": "Separate verifier head", "unified": "Unified head"}
LINKS = {
    "01": "01_chartqa_original_study_report.html",
    "02": "02_multidataset_qhead_report.html",
    "03": "03_qhead_comparison_tables.md",
    "05": "05_mixture_new_loss_report.html",
    "06": "06_corruption_new_loss_report.html",
    "07": "07_correction_experiments_report.html",
}

EXTRA_CSS = """
.callout{border-left:4px solid var(--uj);background:var(--soft);border-radius:0 10px 10px 0;padding:10px 16px;margin:16px 0}
.callout.warn{border-left-color:var(--sp)}.callout b.t{display:block;font-size:.8rem;letter-spacing:.04em;text-transform:uppercase;color:var(--ink2);margin-bottom:2px}
.eq{margin:14px 0;overflow-x:auto}.btn{display:inline-block;margin:10px 0 0;padding:8px 16px;border:1px solid var(--axis);border-radius:8px;background:var(--surface);color:var(--ink);font:inherit;cursor:pointer}
.chip{display:inline-block;font-size:.78rem;padding:1px 8px;border-radius:999px;background:var(--soft);color:var(--ink2);margin-right:4px}
.cover{padding:8px 0 4px}.cover h1{font-size:2.3rem}.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
.kpi div{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 14px}.kpi b{display:block;font-size:1.5rem;line-height:1.2}.kpi span{font-size:.85rem;color:var(--ink2)}
ul{list-style:none;padding-left:20px}ul>li{position:relative}ul>li::before{content:"•";position:absolute;left:-15px;color:var(--ink2)}
.tag{font-size:.8rem;color:var(--muted)}td.num,th.num{text-align:right}.refs li{font-size:.9rem}
h2{break-after:avoid}h3,h4{break-after:avoid}
@media(max-width:700px){.kpi{grid-template-columns:1fr 1fr}}
@page{size:Letter;margin:15mm}
@media print{
 :root{color-scheme:light}body{background:#fff;font-size:10.5pt;line-height:1.5;-webkit-print-color-adjust:exact;print-color-adjust:exact}
 main{max-width:none;padding:0}.toc,.btn,.noprint{display:none!important}
 h2{margin-top:22px}
 pre{white-space:pre-wrap;word-break:break-word}
 figure,.card,.callout,.case,tr,.kpi div,svg{break-inside:avoid}table{font-size:8.6pt}th,td{padding:4px 6px}
 a{color:inherit;text-decoration:none}.tablewrap{overflow:visible}
}
"""

KATEX_START = """
<script>
window.addEventListener('DOMContentLoaded',function(){
  if(window.renderMathInElement){
    renderMathInElement(document.body,{delimiters:[{left:'\\\\[',right:'\\\\]',display:true},{left:'\\\\(',right:'\\\\)',display:false}],throwOnError:false});
  }
  document.documentElement.setAttribute('data-math','done');
});
</script>
"""


# ------------------------------------------------------------------ data helpers


def load_json(path: Path):
    """Read one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def fill(template: str, **values) -> str:
    """Replace @@name@@ markers in a text block, so LaTeX braces never clash with formatting."""
    def sub(match):
        name = match.group(1)
        if name not in values:
            raise KeyError(f"missing value for @@{name}@@")
        return str(values[name])
    return re.sub(r"@@(\w+)@@", sub, template)


def params(kind: str, arch: str, d: int = FEATURE) -> int:
    """Exact parameter count of a head, from its layer sizes."""
    hidden = {"mlp_256": 256, "mlp_512": 512}.get(arch)
    if kind in ("pre", "verifier"):
        return d + 1 if hidden is None else 2 * d + d * hidden + hidden + hidden + 1
    norms = 4 * d
    return norms + (2 * d + 1) + 1 if hidden is None else norms + (2 * d + 1) * hidden + hidden + hidden + 1


def mb(count: int) -> str:
    """Parameter count as megabytes of float32 weights."""
    return f"{count * 4 / 1e6:.2f}"


def fmt(value: float, digits: int = 3) -> str:
    """A number to a fixed number of decimals, or n/a."""
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def pm(pair, digits: int = 3) -> str:
    """A (mean, std) pair as 'mean ± std', or n/a."""
    return "n/a" if pair is None else f"{pair[0]:.{digits}f} ± {pair[1]:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    """A fraction as a percentage string."""
    return f"{100 * value:.{digits}f}%"


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of a proportion."""
    p = correct / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - spread) / denom, (centre + spread) / denom


def bar_chart(rows, xmin: float, xmax: float, xlabel: str, caption_id: str, fmt_value=lambda v: f"{v:.3f}") -> str:
    """Horizontal bars, one row per item: (label, value, css class, optional marker value)."""
    row_h, left, top, width = 30, 230, 8, 640
    height = top + row_h * len(rows) + 40
    scale = (width - left - 60) / (xmax - xmin)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{caption_id}">']
    for i in range(6):
        tick = xmin + i * (xmax - xmin) / 5
        x = left + (tick - xmin) * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + row_h * len(rows)}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top + row_h * len(rows) + 15}" text-anchor="middle">{tick:g}</text>')
    parts.append(f'<text class="axlabel" x="{left + (width - left - 60) / 2}" y="{top + row_h * len(rows) + 33}" text-anchor="middle">{esc(xlabel)}</text>')
    for index, (label, value, css, marker) in enumerate(rows):
        yc = top + row_h * index + row_h / 2
        parts.append(f'<text class="rowlabel" x="{left - 10}" y="{yc + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        x1 = left + (value - xmin) * scale
        parts.append(f'<rect class="{css} bar" x="{left}" y="{yc - 9:.1f}" width="{max(1, x1 - left):.1f}" height="18" rx="3"><title>{esc(label)}: {fmt_value(value)}</title></rect>')
        parts.append(f'<text class="tick" x="{x1 + 5:.1f}" y="{yc + 4:.1f}">{fmt_value(value)}</text>')
        if marker is not None:
            xm = left + (marker - xmin) * scale
            parts.append(f'<line class="chance" x1="{xm:.1f}" x2="{xm:.1f}" y1="{yc - 12:.1f}" y2="{yc + 12:.1f}"><title>Upper bound {fmt_value(marker)}</title></line>')
    parts.append("</svg>")
    return "".join(parts)


BAR_CSS = ".bar{fill:var(--c)}"


# ------------------------------------------------------------------ per-dataset data


def per_dataset() -> dict:
    """Load the per-dataset sweeps and analyses of the four-dataset study."""
    out = {}
    for key, name in DATASETS:
        analysis = load_json(SWEEP / f"{key}_analysis.json")
        separate = load_json(SWEEP / f"{key}_separate_heads_sweep.json")
        unified = load_json(SWEEP / f"{key}_unified_head_sweep.json")
        summary = unified["test_summary"]
        metrics = {
            "token_confidence": analysis["token_confidence_metrics"],
            "separate_pre_generation": separate["pre_generation"]["test"],
            "separate_verifier": separate["candidate_verifier"]["test"],
            "unified_pre_generation": {m: v["mean"] for m, v in summary["pre_generation"].items()},
            "unified_joint": {m: v["mean"] for m, v in summary["joint"].items()},
        }
        out[key] = {"name": name, "analysis": analysis, "separate": separate, "unified": unified, "metrics": metrics}
    return out


def capture_at(analysis: dict, scorer: str, flag: float = 0.2) -> float:
    """Share of all errors that sit in the lowest-scored fraction of answers."""
    return analysis["scorers"][scorer]["error_capture"][analysis["grid"].index(flag)]


def accuracy_at(analysis: dict, scorer: str, coverage: float = 0.5) -> float:
    """Accuracy of the most trusted fraction of answers."""
    return analysis["scorers"][scorer]["accuracy_at_coverage"][analysis["grid"].index(coverage)]


# ------------------------------------------------------------------ sections


def cover(r: Results, pd: dict, kpis: list) -> str:
    """Title block, key numbers, and the export button."""
    cards = "".join(f"<div><b>{value}</b><span>{label}</span></div>" for value, label in kpis)
    return f"""<div class="cover"><h1>Approach 1: Latent Self-Correction, From Detection to Correction</h1>
<p class="sub">A complete account of the Approach 1 work on Qwen3-VL-2B: what was built, why each step was taken, and what every experiment showed. Written for the mid-point report and presentation.</p>
<p class="meta">Model: Qwen3-VL-2B (frozen). Datasets: ChartQA, TextVQA, DocVQA, ScienceQA. Every metric is on held-out test splits unless stated. Detailed per-experiment reports are linked in the text and listed in Appendix A.</p>
<button class="btn" onclick="window.print()">Save as PDF (print this page)</button>
<div class="kpi">{cards}</div></div>"""


def toc(items) -> str:
    """The table of contents."""
    return '<nav class="toc">' + "".join(f'<a href="#{i}">{t}</a>' for i, t in items) + "</nav>"


def section_overview(r: Results) -> str:
    """Section 1: the goal, the two halves, and the stage-by-stage roadmap."""
    rows = [
        ["1", "Baseline: answer-token probability", "See whether the model's own confidence can tell right answers from wrong ones", "AUROC 0.750 on ChartQA and 0.719 pooled. It equals 1.0 on 77% to 90% of answers", "01, 02"],
        ["2", "Original ChartQA study", "Train small heads on frozen hidden states: a pre-generation head and a verifier", "AUROC 0.89, error AUPRC 0.74, 97% accuracy on the most trusted half", "01"],
        ["3", "Three more datasets", "Check the result is not a ChartQA artifact: TextVQA, DocVQA, ScienceQA", "Heads beat the answer-token probability by 0.13 to 0.27 AUROC on every dataset", "02"],
        ["4", "Unified head", "One network for both jobs: score before the answer and after it", "Matches or beats the separate heads on every dataset", "02"],
        ["5", "Pooled mixture", "Train and test on all four datasets at once, old loss", "Unified head AUROC 0.881 against 0.719 for the answer-token probability", "05"],
        ["6", "New loss, swapped image", "Add a visual-grounding term so the score falls when the image is wrong", "Grounding rises from 93% to 96% with no loss of AUROC", "05"],
        ["7", "New loss, noised image", "Same loss, but corrupt the image by adding noise instead of swapping it", "Grounding rises from 92% to 99% and carries over to other corruptions", "06"],
        ["8", "Sequential training", "Train on A, then B, then C, and check A is still remembered", "Forgetting 0.011 to 0.046 AUROC depending on head and loss; replay removes most of it for the verifier", "05, 06"],
        ["9", "Trained heads exported", "Save every trained head so teammates can build on it", "540 heads, one to eight megabytes each", "this report"],
        ["10", "Correction experiments", "Use the head to pick or revise answers (retry, best-of-five, deeper pass)", "Head selection among five samples: +0.4 points over greedy on the full test split (not significant); the head ranks candidates well (AUROC 0.883 against 0.815)", "07"],
    ]
    return f"""<h2 id="overview" class="first">1. The goal and the road map</h2>
<p class="lead">Approach 1 asks a vision-language model to notice when its own visual evidence is weak and to act on that before answering. It has two halves.</p>
<p><b>Detection (the gate).</b> A small network reads the hidden state of the frozen model and outputs the probability that the answer is correct. If this works, the model can be told when to slow down, retry, or hand the question to something else.</p>
<p><b>Correction.</b> A step that acts on answers the gate flags. Everything in Sections 3 to 11 builds and tests the gate. Section 12 reports the first experiments on acting on it.</p>
{figure("fig-pipeline", "Where the correctness head sits", pipeline_diagram(), "The language model is frozen throughout. Only the small head is trained. The correction stage on the right is where the experiments of Section 12 apply.")}
<h3>1.1 The ten stages, in the order they happened</h3>
{table(["#", "Stage", "What was done and why", "Main result", "Report"], rows, left_cols=(1, 2, 3))}
<p class="small">Report numbers refer to the files in <code>results/approach1/</code>: 01 original ChartQA study, 02 multi-dataset report, 03 metric tables, 05 mixture and new loss, 06 noise corruption, 07 correction experiments.</p>
<div class="callout"><b class="t">Terms used throughout</b>The <b>answer-token probability</b> is the untrained baseline: the probability the model itself gave its answer tokens. A <b>linear probe</b> is a trained head with no hidden layer. A <b>head</b> is any small trained network that reads the frozen hidden states. <b>AUROC</b> is the chance that a randomly chosen correct answer gets a higher score than a randomly chosen wrong one.</div>"""


def section_foundations(r: Results, pd: dict) -> str:
    """Section 2: model, features, labels, splits, metrics, calibration."""
    rows = []
    for key, name in DATASETS:
        a = pd[key]["analysis"]["splits"]
        rows.append([name, f"{a['train']['count']:,}", f"{a['val']['count']:,}", f"{a['test']['count']:,}", pct(a["train"]["accuracy"]), pct(a["test"]["accuracy"]), RULE[key]])
    text = r"""
<h2 id="foundations">2. Foundations: what is measured and how</h2>
<h3>2.1 The model and what is cached</h3>
<p>The vision-language model is Qwen3-VL-2B, kept frozen for the whole project. It answers every question once. For each question the last-layer hidden state of the last token is cached in three situations, so that all later experiments reuse them and never run the large model again.</p>
<ul>
<li><b>Prompt state</b> \(h_q\): the image and the question, ending at the generation prompt. It exists before any answer does.</li>
<li><b>Answer state</b> \(h_a\): the image, the question, and the model's own answer appended. It describes a specific candidate answer.</li>
<li><b>Reference state</b> \(h_{ref}\): the same, with the correct reference answer appended. Used only as training supervision.</li>
</ul>
<p>Each state is a vector of size \(d = @@d@@\). Caching is what makes the experiments cheap. Extracting the states of one dataset split takes up to an hour and a half on one GPU (82 minutes for the 28,299 ChartQA training questions with a corrupted image), but a whole pooled sweep of head fits on the cached states takes 3 to 12 minutes per head kind. The price is that nothing the head learns can change what the model sees or says.</p>
<h3>2.2 What counts as a correct answer</h3>
<p>Each dataset has its own rule, converted to one binary label \(y \in \{0,1\}\). Correctness is judged by the dataset's official scoring, not by exact string match, because exact match would call a right answer wrong whenever formatting differs.</p>
@@dtable@@
<p>ChartQA uses relaxed accuracy (a numeric answer within 5% of the reference). TextVQA uses the VQA soft accuracy over ten human answers, correct at 0.5 or more. DocVQA uses ANLS, correct at 0.9 or more. ScienceQA uses the option letter. Splits are grouped by image or document so that no image appears in two splits, and the validation and test splits of TextVQA and DocVQA are carved out of the official validation split because the official test answers are hidden.</p>
<h3>2.3 Metrics</h3>
<p>The head outputs a score \(s\) and the probability \(p = \sigma(s / T)\), where \(\sigma\) is the logistic function and \(T\) is a temperature fitted once on the validation split by minimizing the cross-entropy (temperature scaling, Guo et al. 2017). The test split is used only for the final numbers.</p>
<ul>
<li><b>Correctness AUROC</b>, \(\mathrm{AUROC} = P(s_{\text{correct}} &gt; s_{\text{wrong}})\), the probability that a random correct answer outranks a random wrong one. 0.5 is a coin flip, 1.0 is perfect.</li>
<li><b>Error AUPRC</b>, the area under the precision-recall curve when the <em>wrong</em> answers are the positive class. It measures how well errors are packed at the top of a queue sorted from least to most trusted, which is the view that matters for a gate. Chance equals the error rate.</li>
<li><b>Brier score</b>, \(\frac{1}{n}\sum_i (p_i - y_i)^2\). Lower is better. It rewards being right and being honest about uncertainty.</li>
<li><b>ECE</b>, the expected calibration error, \(\sum_b \frac{|B_b|}{n}\,\bigl|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\bigr|\) over ten probability bins. Lower is better.</li>
<li><b>Accuracy at coverage</b>: keep the most trusted fraction of answers and report their accuracy. Full coverage equals the base accuracy.</li>
<li><b>Errors captured at a flag rate</b>: flag the lowest-scored fraction of answers and report the share of all errors that fall inside.</li>
</ul>
<h3>2.4 Selection protocol</h3>
<p>Every configuration is trained on three seeds. A winner is the configuration with the best mean validation error AUPRC (validation AUROC breaks ties), chosen on the validation split only. Test numbers are read afterwards for the winners. Where a number is shown with a plus-minus sign, it is the standard deviation across seeds. This matters because several differences in this project are a few thousandths of AUROC, which is the size of the seed noise.</p>
"""
    return fill(text, d=FEATURE, dtable=table(["Dataset", "Train", "Validation", "Test", "Base accuracy, train", "Base accuracy, test", "Correct means"], rows, left_cols=(0, 6)))


def section_baseline(r: Results, pd: dict) -> str:
    """Section 3: the answer-token probability and why it fails."""
    rows = []
    for key, name in DATASETS:
        a = pd[key]["analysis"]
        t = a["token_confidence_metrics"]
        rows.append([name, pct(a["splits"]["test"]["accuracy"]), pct(a["token_confidence_exactly_one"]), fmt(t["correctness_auroc"]), fmt(t["error_auprc"]), fmt(t["brier"]), fmt(t["ece_10_bin"])])
    pooled = r.baseline("pooled")["token_confidence"]
    rows.append(["Pooled mixture", pct(r.analysis["composition"]["test"]["accuracy"]), pct(r.baseline("pooled")["exactly_one"]), fmt(pooled["correctness_auroc"]), fmt(pooled["error_auprc"]), fmt(pooled["brier"]), fmt(pooled["ece_10_bin"])])
    text = r"""
<h2 id="baseline">3. Stage 1: the model's own confidence</h2>
<p>The natural first idea is to use what the model already says about itself. When Qwen3-VL generates an answer of \(m\) tokens, it assigns each token a probability. The geometric mean of those probabilities is a confidence for the whole answer:</p>
<div class="eq">\[ c = \exp\!\Bigl(\tfrac{1}{m}\sum_{j=1}^{m}\log p(t_j \mid t_{&lt;j},\ \text{image},\ \text{question})\Bigr) \]</div>
<p>It needs no training and no extra computation. We call it the <b>answer-token probability</b> throughout. Its problem is saturation: modern instruction-tuned models are very sure of their tokens whether or not the answer is right, so the score is exactly 1.0 for most answers and cannot rank them.</p>
@@table@@
<div class="callout"><b class="t">Why this stage matters</b>On the original ChartQA study the score was exactly 1.0 for 1,921 of 2,500 test answers (76.8%), including many wrong ones. Keeping its most confident half of answers raised accuracy only from 75.8% to 92.4%. Across the four datasets the score sits at 1.0 for 77% to 90% of answers, so a threshold on it cannot separate right from wrong. This is the gap the learned heads are built to close, and it is the comparison every later result is measured against.</div>
<p>An answer-length baseline (does a longer answer tend to be wrong?) was also computed as a sanity check on shortcuts. It reaches an AUROC of only 0.52 on the pooled mixture, so the heads below are not simply learning answer length.</p>
"""
    return fill(text, table=table(["Dataset", "Base accuracy", "Token probability exactly 1.0", "AUROC", "Error AUPRC", "Brier", "ECE"], rows))


def section_original(r: Results) -> str:
    """Section 4: the original ChartQA study."""
    runs = load_json(SWEEP / "chartqa_original_validation_runs.json")
    groups = {}
    for run in runs:
        groups.setdefault((run["head"], run["architecture"], run["ranking_weight"]), []).append(run["auroc"])
    sweep_rows = []
    for (head, arch, rank), values in sorted(groups.items()):
        sweep_rows.append(["Pre-generation" if head == "pre_generation" else "Verifier", arch, f"{rank:g}", f"{params('pre' if head == 'pre_generation' else 'verifier', arch):,}", f"{sum(values) / len(values):.4f}", f"{min(values):.4f} to {max(values):.4f}"])
    columns = ["Metric", "Answer-token probability", "Pre-gen MLP", "Pre-gen linear probe", "Verifier MLP (swept)", "Verifier, early stop (3 seeds)"]
    test_rows = [[metric] + values for metric, values in ORIGINAL_TEST.items()]
    text = r"""
<h2 id="original">4. Stage 2: the original ChartQA study</h2>
<p><b>Question.</b> Before building anything that corrects a model, can the model's internal state tell us when it is about to be wrong? This was first tested on ChartQA alone. The full study is report <a href="@@l01@@">01</a>.</p>
<h3>4.1 Two heads, two questions</h3>
<ul>
<li>The <b>pre-generation head</b> reads \(h_q\) and predicts whether the question will be answered correctly before any answer exists. It measures how hard the question looks to the model.</li>
<li>The <b>verifier head</b> reads \(h_a\) and judges one specific generated answer. This is the head a correction step needs, because it scores the candidate itself.</li>
</ul>
<p>Both are small networks trained on a correct or wrong label, with the language model frozen. The reason for two heads is that they answer different questions at different times: one before the answer (useful to decide whether to generate more carefully), one after (useful to decide whether to keep or revise).</p>
<h3>4.2 Architecture and training</h3>
<p>The default head is a two-layer network: layer normalization, a linear layer to 512 units, GELU, 10% dropout, one output. It was trained with AdamW (learning rate 0.001, weight decay 0.0001, batch size 512, 20 epochs). The later sweep also tested a plain <b>linear probe</b> \(s = w^{\top}h + b\) and a 256-unit variant. The parameter counts are in Section 5.</p>
<h3>4.3 Loss functions of the original study</h3>
<p>Both heads use binary cross-entropy plus a ranking loss. The verifier adds a paired loss that uses the reference answer.</p>
<div class="eq">\[ L_{\mathrm{BCE}} = -\bigl[y\log\sigma(s) + (1-y)\log(1-\sigma(s))\bigr] \]
\[ L_{\mathrm{rank}} = \operatorname*{mean}_{i\in\text{correct},\ j\in\text{wrong}} \operatorname{softplus}\bigl(-(s_i - s_j)\bigr),\qquad \operatorname{softplus}(z)=\log(1+e^{z}) \]
\[ L_{\mathrm{pair}} = \operatorname*{mean}_{j\in\text{wrong}} \operatorname{softplus}\bigl(-(s_{\mathrm{ref},j} - s_{\mathrm{cand},j})\bigr) \]
\[ L_{\mathrm{pre}} = L_{\mathrm{BCE}} + 0.25\,L_{\mathrm{rank}},\qquad L_{\mathrm{verifier}} = L_{\mathrm{BCE}} + 0.25\,L_{\mathrm{rank}} + 0.25\,L_{\mathrm{pair}} \]</div>
<p>The ranking term compares every correct example with every wrong one in a batch. The paired term compares the model's wrong answer with the correct reference answer for the same image and question, using two teacher-forced passes that differ only in the answer text. The reference is training supervision only: at deployment the verifier sees the image, question and candidate, never the unknown correct answer. The idea was to teach the verifier what a right answer looks like by contrast.</p>
<h3>4.4 The sweep</h3>
<p>Using cached features, 12 configurations per head, each on three seeds (36 runs per head, 72 in total), were trained: architecture \(\in\) {linear, MLP-256, MLP-512}, learning rate \(\in\) {3e-4, 1e-3}, ranking weight \(\in\) {0, 0.25}, seed \(\in\) {1, 2, 3}. A second sweep added early stopping and five ranking weights for the verifier. The table gives mean validation AUROC of the 72 runs, grouped. The raw runs are in <code>04_sweep_data/chartqa_original_validation_runs.json</code>.</p>
@@sweep@@
<h3>4.5 Results (test split, 2,500 questions)</h3>
@@test@@
<div class="callout"><b class="t">What the study found</b>(1) The answer-token probability is weak (AUROC 0.750). (2) A small head on frozen states is far better: AUROC 0.89, error AUPRC about 0.74, Brier score 0.178 to 0.11, ECE 0.172 to between 0.03 and 0.04, and accuracy on the most trusted half of answers from 92.4% to about 97%. (3) The linear probe had the highest AUROC (0.895), which says expected difficulty is close to linearly readable from the question state. The early-stopped verifier had the highest error AUPRC (0.738) and is the relevant model for judging a specific answer. (4) The ranking objectives did not help: plain cross-entropy won both heads in the main sweep.</div>
<p><b>Why these choices were made.</b> Linear probes were included because a linear probe that works is the strongest evidence that the information is already in the hidden state, and not created by the head. The sweep was run on cached features so that the choice of architecture, learning rate and ranking weight rests on 72 fits and not on one lucky run. Selection used validation error AUPRC because for a gate, finding the wrong answers matters more than the average ranking.</p>
"""
    return fill(text, l01=LINKS["01"], sweep=table(["Head", "Architecture", "Ranking weight", "Parameters", "Mean validation AUROC", "Range over runs"], sweep_rows, left_cols=(0, 1)), test=table(columns, test_rows))


def section_sizes(r: Results, pd: dict) -> str:
    """Section 5: head sizes and what was selected everywhere."""
    vlm = 2.0e9
    rows = []
    for kind in KINDS:
        for arch in ("linear", "mlp_256", "mlp_512"):
            n = params(kind, arch)
            rows.append([KIND_NAME[kind], arch, f"{FEATURE if kind != 'unified' else 2 * FEATURE:,}", f"{n:,}", mb(n), f"{100 * n / vlm:.3f}%"])
    chart = bar_chart([(f"{KIND_NAME[k].replace('Separate ', '')}, {a}", params(k, a) / 1e6, "c-uj" if k == "unified" else "c-sv" if k == "verifier" else "c-sp", None) for k in KINDS for a in ("linear", "mlp_256", "mlp_512")],
                      0, 2.4, "Parameters in millions", "fig-size", lambda v: f"{v:.3f} M")
    cfg_rows = []
    for key, name in DATASETS:
        s, u = pd[key]["separate"], pd[key]["unified"]
        pre_c, ver_c, uni_c = s["pre_generation"]["best_config"], s["candidate_verifier"]["best_config"], u["winner"]["config"]
        cfg_rows.append([name,
                         f"{pre_c['architecture']} ({params('pre', pre_c['architecture']):,})",
                         f"{ver_c['architecture']} ({params('verifier', ver_c['architecture']):,})",
                         f"{uni_c['architecture']} ({params('unified', uni_c['architecture']):,})"])
    pooled_rows = []
    for kind in KINDS:
        for group, label in (("old", "old loss"), ("new", "new loss")):
            w = r.winner(kind, group)
            if w:
                c = w["config"]
                pooled_rows.append([KIND_NAME[kind], label, c["architecture"], f"{c['learning_rate']:g}", f"{params(kind, c['architecture']):,}", f"{c['ground_weight']:g}", f"{c['pair_weight']:g}"])
    text = r"""
<h2 id="sizes">5. Head sizes</h2>
<p>The size of the heads matters for two reasons: it bounds the cost of using them (they add almost nothing to inference) and it shows that the signal is cheap to read. The counts below are exact, computed from the layer sizes for the 2,048-dimensional hidden state of Qwen3-VL-2B. For scale, the language model has roughly 2 billion parameters, so the largest head here is about 0.1% of it.</p>
<h3>5.1 Architectures</h3>
<ul>
<li><b>Linear probe</b> (pre-generation or verifier): \(s = w^{\top}h + b\), with \(d + 1\) parameters.</li>
<li><b>MLP-\(k\)</b> (pre-generation or verifier): LayerNorm \((2d)\), Linear \(d \to k\), GELU, Dropout 0.1, Linear \(k \to 1\). The parameter count is \(2d + dk + k + k + 1\).</li>
<li><b>Unified head</b>: a separate LayerNorm on each half (\(4d\) parameters), then the concatenation \([\,\mathrm{LN}(h_q)\ \|\ \mathrm{LN}(h_a)\ \|\ f\,]\) of size \(2d+1\) (the last entry is a flag), then a linear layer or the same MLP body. The count is \(4d + (2d+1)k + k + k + 1\) for an MLP.</li>
</ul>
@@size_table@@
@@size_chart@@
<p>The unified head is about twice the size of a separate MLP because it reads two states at once. Its 2.11 million parameters are the same as a separate pre-generation head and a verifier of the same width together (2.11 million), so serving one unified head costs about the same as serving the two separate ones.</p>
<h3>5.2 What was chosen in each experiment</h3>
<p>No single architecture wins everywhere. Linear probes win the pre-generation and unified heads on ChartQA and the verifier on DocVQA, and the other winners use a hidden layer. Differences between architectures are usually under 0.01 AUROC, so size is not what decides the result. Parameter counts are in brackets.</p>
@@cfg@@
<p><b>Pooled mixture winners</b> (Section 8). The new loss weights are the grounding weight \(\lambda\) and the pair weight \(w_{\text{pair}}\).</p>
@@pooled@@
"""
    return fill(text, size_table=table(["Head", "Architecture", "Input size", "Parameters", "Megabytes (fp32)", "Share of the 2B model"], rows, left_cols=(0, 1)),
                size_chart=figure("fig-size", "Parameters of every head architecture, in millions", chart, "A separate MLP-512 has about 1.05 million parameters. The unified MLP-512 has 2.11 million. The linear probes are 2,049 to 12,290 parameters."),
                cfg=table(["Dataset", "Pre-generation head", "Verifier head", "Unified head"], cfg_rows, left_cols=(0, 1, 2, 3)),
                pooled=table(["Head", "Loss", "Architecture", "Learning rate", "Parameters", "\\(\\lambda\\)", "\\(w_{pair}\\)"], pooled_rows, left_cols=(0, 1, 2)))


def section_datasets(r: Results, pd: dict) -> str:
    """Section 6: three more datasets and the per-dataset results."""
    cards = ""
    for key, name in DATASETS:
        info = DATASET_INFO[key]
        cards += f"""<div class="card" style="margin:0"><h4 style="margin-top:0">{name}</h4><p><b>What it asks.</b> {esc(info['what'])}</p><p><b>Source.</b> {esc(info['source'])}. <b>Correct means.</b> {esc(info['rule'])}</p><p><b>Why it is here.</b> {esc(info['why'])}</p><p class="small"><b>Splits.</b> {esc(info['splits'])}</p></div>"""
    metric_rows = {}
    for metric, label in (("correctness_auroc", "Correctness AUROC"), ("error_auprc", "Error AUPRC"), ("brier", "Brier score (lower is better)"), ("ece_10_bin", "ECE (lower is better)")):
        rows = []
        for scorer, name, _ in SCORERS:
            rows.append([name] + [fmt(pd[k]["metrics"][scorer][metric]) for k, _ in DATASETS])
        best = max(range(len(rows)), key=lambda i: float(rows[i][1])) if metric in ("correctness_auroc", "error_auprc") else None
        metric_rows[metric] = (label, rows)
    tables = ""
    for metric, (label, rows) in metric_rows.items():
        tables += f"<h4>{label}</h4>" + table(["Scorer"] + [n for _, n in DATASETS], rows)
    chart_rows = [(n, {s: pd[k]["metrics"][s]["correctness_auroc"] for s, _, _ in SCORERS}) for k, n in DATASETS]
    chart = dot_plot(chart_rows, SCORERS, 0.6, 1.0, "Correctness AUROC on the test split (axis starts at 0.6)", caption_id="fig-auroc4")
    gate_rows = []
    for key, name in DATASETS:
        a = pd[key]["analysis"]
        gate_rows.append([name, pct(capture_at(a, "token_confidence"), 0), pct(capture_at(a, "unified_joint"), 0), pct(accuracy_at(a, "token_confidence")), pct(accuracy_at(a, "unified_joint")), pct(a["splits"]["test"]["accuracy"])])
    capture_panels = []
    for key, name in DATASETS:
        a = pd[key]["analysis"]
        capture_panels.append(line_panel(name, a["grid"], [(s, a["grid"], a["scorers"][s]["error_capture"]) for s in ("token_confidence", "separate_verifier", "unified_joint")], 0.0, 1.0, 0.25, "Fraction of answers flagged (least trusted first)", "Share of all errors captured", diagonal=True))
    boot_rows = []
    for key, name in DATASETS:
        b = pd[key]["analysis"]["bootstrap"]["joint_minus_separate_verifier"]["auroc"]
        p = pd[key]["analysis"]["bootstrap"]["unified_pre_minus_separate_pre"]["auroc"]
        boot_rows.append([name, f"{b['difference']:+.4f} ({b['ci_low']:+.4f} to {b['ci_high']:+.4f})", f"{p['difference']:+.4f} ({p['ci_low']:+.4f} to {p['ci_high']:+.4f})"])
    text = r"""
<h2 id="datasets">6. Stage 3: three more datasets</h2>
<p><b>Why.</b> A result on one dataset can be an accident of that dataset. ChartQA questions are templated, the answers are numbers, and the model's mistakes may have a simple structure. If the heads only worked there, the project would have a ChartQA tool and not a general gate. Three datasets were added, chosen because the model has to do something different in each: read text in a photograph, read a scanned page, and reason over a diagram with a closed answer set. Each has its own correctness rule, grouped splits, and an offline-tested scoring module. Full details are in report <a href="@@l02@@">02</a>.</p>
@@cards@@
<h3>6.1 Protocol</h3>
<p>The same recipe was run per dataset: build the grouped splits, generate the baseline answers, cache the three hidden states, run the separate-head sweep (12 configurations per head, three seeds each) and the unified-head sweep (36 configurations, three seeds, early stopping, maximum 50 epochs, patience 5), and analyze the winners. ChartQA was rerun end to end under this protocol as a pipeline check; it reproduced the original numbers (answer-token probability AUROC 0.751 against 0.750, linear probe 0.895 against 0.895).</p>
<h3>6.2 Results</h3>
<p>All numbers are on each dataset's test split. The unified head rows are means over three seeds.</p>
@@tables@@
@@legend@@
@@chart@@
<div class="callout"><b class="t">Reading the numbers</b>The learned heads beat the answer-token probability on every dataset and every metric. The gain in AUROC of the unified joint head runs from 0.13 (DocVQA) to 0.27 (ScienceQA). Calibration improves most: ECE falls from the range 0.11 to 0.18 for the probability to between 0.01 and 0.04 for the heads, so the head's number can be read as a probability. The separate verifier and the unified joint head are close everywhere. The pre-generation scores sit lower than the answer-aware ones on TextVQA and DocVQA, where the answer text carries the signal, and close to them on ScienceQA, where the question alone tells the head which topics are hard.</div>
<h3>6.3 Using the score as a gate</h3>
<p>If a correction step can only act on a fraction of answers, what matters is how many errors the flagged fraction contains. Flagging the 20% of answers with the lowest score:</p>
@@gate@@
@@captures@@
<p>The dashed diagonal is random flagging. The learned head's curve sits well above the answer-token probability on every dataset.</p>
<h3>6.4 Does the head have to be a pair?</h3>
<p>Paired bootstrap on the test split (1,000 resamples) gives the difference between the unified head and the separate head of the same kind, as AUROC with a 95% interval. An interval that includes zero means the two are not distinguishable.</p>
@@boot@@
<p>In joint mode the unified head is never worse than the separate verifier, and its interval excludes zero in its favor on DocVQA and ScienceQA (on ChartQA and TextVQA the interval includes zero). In pre-generation mode it is within 0.006 AUROC of the separate head: slightly better on ScienceQA, slightly lower on DocVQA, and indistinguishable on the other two. This is why the unified head is kept: it matches two separate heads on the whole and it is one network.</p>
"""
    return fill(text, l02=LINKS["02"], cards=f'<div class="grid2">{cards}</div>', tables=tables, legend="",
                chart=figure("fig-auroc4", "Correctness AUROC by dataset and scorer", chart + legend([(label, css) for _, label, css in SCORERS]), "Each row is one dataset. The axis starts at 0.6."),
                gate=table(["Dataset", "Errors found, answer-token probability", "Errors found, unified head", "Accuracy of top half, answer-token probability", "Accuracy of top half, unified head", "Base accuracy"], gate_rows),
                captures=f'<div class="grid2">{"".join(capture_panels)}</div>', boot=table(["Dataset", "Unified joint minus separate verifier", "Unified pre-generation minus separate pre-generation"], boot_rows))


def section_unified(r: Results, pd: dict) -> str:
    """Section 7: the unified head."""
    rows = []
    for key, name in DATASETS:
        m = pd[key]["metrics"]
        rows.append([name, fmt(m["separate_pre_generation"]["correctness_auroc"]), fmt(m["unified_pre_generation"]["correctness_auroc"]), fmt(m["separate_verifier"]["correctness_auroc"]), fmt(m["unified_joint"]["correctness_auroc"])])
    text = r"""
<h2 id="unified">7. Stage 4: the unified head</h2>
<p><b>The idea.</b> Two separate heads need two networks, two trainings and two decisions about which to trust. A single head that reads both states removes that. It also gives a natural way to combine them: the question state \(h_q\) gives the difficulty prior, and the answer state \(h_a\) gives the evidence about this particular answer.</p>
@@diagram@@
<h3>7.1 Design</h3>
<p>Each half is normalized separately, so neither dominates by scale. The input is the concatenation and a flag \(f\):</p>
<div class="eq">\[ x = \bigl[\ \mathrm{LN}_q(h_q)\ \|\ \mathrm{LN}_a(h_a)\ \|\ f\ \bigr],\qquad s_{\mathrm{joint}} = g(x),\ f=1 \]
\[ s_{\mathrm{pre}} = g\bigl(\bigl[\ \mathrm{LN}_q(h_q)\ \|\ \mathbf{0}\ \|\ 0\ \bigr]\bigr) \]</div>
<p>With the answer half zeroed and the flag at 0, the same network scores a question before any answer exists (<b>pre-generation mode</b>). With both halves present it scores a candidate answer (<b>joint mode</b>). This is trained jointly with a loss that keeps both modes useful:</p>
<div class="eq">\[ L = L_{\mathrm{BCE}}(s_{\mathrm{joint}}, y) + w_{\mathrm{pre}}\,L_{\mathrm{BCE}}(s_{\mathrm{pre}}, y) + w_{\mathrm{rank}}L_{\mathrm{rank}} + w_{\mathrm{pair}}L_{\mathrm{pair}} \]</div>
<p>The reason for the pre-generation term is that a head trained only on the joint input would learn to rely on the answer half and could be useless when the answer half is empty. The flag and the zeroing teach it a separate, honest mode for the question-only case.</p>
<h3>7.2 Results</h3>
<p>Test AUROC per dataset, separate heads against the unified head in the same mode.</p>
@@table@@
<div class="callout"><b class="t">Findings</b>The unified head's pre-generation mode matches the separate pre-generation head to within 0.007 AUROC on every dataset, and its joint mode matches or beats the separate verifier on all four datasets (the largest gains are on DocVQA and ScienceQA). The pre-generation loss term costs almost nothing: it lifts the question-only mode on every dataset and moves the joint mode by at most 0.006 AUROC on validation. The pairwise reference-answer loss lowered validation AUROC on every dataset and is switched off in all winning configurations. One network therefore does the job of two without a measurable loss.</div>
"""
    return fill(text, diagram=figure("fig-unified", "The unified head", unified_diagram(), "Two normalized halves, one flag, and one output. With the answer half zeroed it is a pre-generation scorer."),
                table=table(["Dataset", "Separate pre-generation", "Unified, pre-generation mode", "Separate verifier", "Unified, joint mode"], rows))


def section_mixture(r: Results) -> str:
    """Section 8: the pooled mixture, old loss."""
    datasets = r.datasets
    labels = ["Pooled"] + [NAMES.get(d, d) for d in datasets]
    keys = [None] + datasets
    rows = []
    base = [r.baseline("pooled" if k is None else k) for k in keys]
    rows.append(["Answer-token probability"] + [fmt(b["token_confidence"]["correctness_auroc"]) for b in base])
    for kind, mode, label in (("pre", "main", "Linear probe, pre-generation"), ("verifier", "main", "Linear probe, verifier"), ("unified", "joint", "Linear probe, unified joint")):
        rows.append([label] + [fmt((r.metric(kind, "old_linear", mode, "correctness_auroc", k) or (float("nan"),))[0]) for k in keys])
    for kind, mode, label in (("pre", "main", "Separate pre-generation head"), ("verifier", "main", "Separate verifier head"), ("unified", "pre_generation", "Unified, pre-generation mode"), ("unified", "joint", "Unified, joint mode")):
        rows.append([label] + [fmt((r.metric(kind, "old", mode, "correctness_auroc", k) or (float("nan"),))[0]) for k in keys])
    spec_rows = []
    for key, name in DATASETS:
        spec = r.specialists.get(key)
        mixture = r.metric("unified", "old", "joint", "correctness_auroc", key)
        if spec and mixture:
            spec_rows.append([name, fmt(spec["unified_joint"]), fmt(mixture[0]), f"{mixture[0] - spec['unified_joint']:+.3f}"])
    chart_rows = [("Pooled", {"token_confidence": base[0]["token_confidence"]["correctness_auroc"], "unified_joint": r.metric("unified", "old", "joint", "correctness_auroc")[0]})]
    for index, key in enumerate(datasets, start=1):
        chart_rows.append((NAMES.get(key, key), {"token_confidence": base[index]["token_confidence"]["correctness_auroc"], "unified_joint": r.metric("unified", "old", "joint", "correctness_auroc", key)[0]}))
    chart = dot_plot(chart_rows, [s for s in SCORERS if s[0] in ("token_confidence", "unified_joint")], 0.6, 1.0, "Correctness AUROC (axis starts at 0.6)", caption_id="fig-mix")
    comp = r.analysis["composition"]
    text = r"""
<h2 id="mixture">8. Stage 5: one head for all four datasets</h2>
<p><b>Why.</b> The per-dataset heads show that the signal exists in each dataset. A deployed gate does not know which dataset a question comes from. The pooled experiment asks whether one head, trained on a mix, scores all of them well, and whether it pays a price against heads trained on a single dataset. Details are in report <a href="@@l05@@">05</a>.</p>
<p><b>Setup.</b> The mixture is the union of the four datasets' existing splits: @@train@@ training rows, @@val@@ validation rows, @@test@@ test rows, with an overall base accuracy of @@acc@@ on the test split. Nothing was resampled, so every test question is one the per-dataset study also scored. One calibration temperature is fitted on the pooled validation split. The sweep has 36 head configurations for the pre-generation head and for the verifier and 72 for the unified head, each on three seeds, covering the old loss (no grounding term) and the new loss.</p>
<h3>8.1 Experiment 1: the answer-token probability and linear probes</h3>
<p>No training is needed for the probability. The linear probes are the best linear heads of the sweep, trained on the pool.</p>
<h3>8.2 Experiments 2 and 3: separate and unified heads, old loss</h3>
@@table@@
@@chart@@
<h3>8.3 Generalist against specialist</h3>
<p>Is a head trained on a mixture worse on a dataset than a head trained on that dataset alone? The table compares the unified joint head. A positive difference means the pooled head is better.</p>
@@spec@@
<div class="callout"><b class="t">Findings</b>One unified head trained on the pool reaches pooled AUROC @@uj@@ against @@tok@@ for the answer-token probability. On each dataset it is within 0.015 AUROC of the head trained on that dataset alone, and above it on three of the four, so the pool does not cost much accuracy and on three datasets helps slightly, possibly because the other datasets add training data for the same signal. A linear probe already reaches 0.86 to 0.88 pooled, so most of the signal is linearly readable; the hidden layer and the answer half add the remaining few points.</div>
"""
    return fill(text, l05=LINKS["05"], train=f"{comp['train']['rows']:,}", val=f"{comp['val']['rows']:,}", test=f"{comp['test']['rows']:,}", acc=pct(comp["test"]["accuracy"]),
                table=table(["Correctness AUROC"] + labels, rows), chart=figure("fig-mix", "Answer-token probability and unified head on the pooled mixture", chart + legend([(label, css) for _, label, css in SCORERS if label in ("Answer-token probability", "Unified, joint mode")]), ""),
                spec=table(["Dataset", "Trained on that dataset alone", "Trained on the pool", "Difference"], spec_rows),
                uj=fmt(r.metric("unified", "old", "joint", "correctness_auroc")[0]), tok=fmt(base[0]["token_confidence"]["correctness_auroc"]))


def section_losses(r: Results, swap: dict, noise: dict, xeval: dict) -> str:
    """Section 9: the old loss, the new loss, swap and noise corruption."""
    mode = {"pre": "main", "verifier": "main", "unified": "joint"}
    rows = []
    for kind in KINDS:
        old, new_s, new_n = swap[kind]["winners"]["old"], swap[kind]["winners"]["new"], noise[kind]["winners"]["new"]
        rows.append([KIND_NAME[kind], fmt(corr.auroc(old, kind), 4), fmt(corr.auroc(new_s, kind), 4), fmt(corr.auroc(new_n, kind), 4),
                     fmt(corr.grounding_against(noise[kind], "old", kind, "noise")), fmt(corr.own_grounding(new_s, kind)), fmt(corr.grounding_against(noise[kind], "new", kind, "noise"))])
    ablation = []
    for kind, label in (("verifier", "Separate verifier"), ("unified", "Unified, joint mode")):
        m = mode[kind]
        for cell, name in (("cell_plain", "Neither term (plain)"), ("cell_contrast", "Pair term only"), ("cell_ground", "Grounding term only"), ("cell_full", "Both terms")):
            if r.winner(kind, cell):
                a = r.metric(kind, cell, m, "correctness_auroc")
                g = r.grounding(kind, cell, m, "order_accuracy")
                ablation.append([label, name, pm(a), pm(g), f"{cell_change(r, kind, m, cell):+.3f}", f"{cell_change(r, kind, m, cell, ground=True):+.3f}"])
    transfer = ""
    for kind in ("verifier", "unified"):
        transfer += f"<h4>{KIND_NAME[kind]}</h4>" + table(["Trained with", "Swap", "Noise", "Mask", "Shuffle"], corr.transfer_rows(xeval[kind], noise[kind], kind), left_cols=(0,))
    plot_rows = []
    for kind in KINDS:
        for corruption in ("mask", "shuffle"):
            plot_rows.append((f"{KIND_NAME[kind].replace('Separate ', '')}, {corruption}", corr.grounding_against(noise[kind], "old", kind, corruption), corr.grounding_against(noise[kind], "new", kind, corruption), None))
    single_rows = []
    for key, name in (("chartqa", "ChartQA"), ("textvqa", "TextVQA"), ("docvqa", "DocVQA"), ("scienceqa", "ScienceQA")):
        swap_s, noise_s = corr.load(f"{key}_swap_unified"), corr.load(f"{key}_noise_unified")
        if swap_s and noise_s:
            single_rows.append([name, fmt(corr.auroc(swap_s["winners"]["old"], "unified")), fmt(corr.auroc(swap_s["winners"]["new"], "unified")), fmt(corr.auroc(noise_s["winners"]["new"], "unified")),
                                fmt(corr.grounding_against(noise_s, "old", "unified", "noise")), fmt(corr.own_grounding(swap_s["winners"]["new"], "unified")), fmt(corr.own_grounding(noise_s["winners"]["new"], "unified"))])
    unified_old = corr.auroc(swap["unified"]["winners"]["old"], "unified")
    text = r"""
<h2 id="losses">9. Loss functions</h2>
<p>Three loss designs were run on the same heads, features and selection protocol, so differences can be attributed to the loss.</p>
<h3>9.1 The old loss</h3>
<p>The old loss asks only for the right label on real inputs (Sections 4 and 7): cross-entropy, with a pre-generation term for the unified head. A head trained this way can reach a high AUROC by learning which kinds of questions the model tends to get wrong, without ever using what the image shows. That is useful for ranking, but it is not what a correction step needs, which is a score that reacts when the visual evidence is missing.</p>
<h3>9.2 The new loss</h3>
<p>The proposed loss has three lines, and each maps to one term:</p>
<div class="eq">\[ P_\theta(S_{\mathrm{conf}} \mid h_{\mathrm{answer}}, h_{\mathrm{question}}) \ \to\ L_{\mathrm{cls}}\ (\text{and } L_{\mathrm{pre}}) \]
\[ \mathcal{L}_{\mathrm{contrastive}}(h_{\mathrm{predicted}}, h_{\mathrm{correct}}) \ \to\ L_{\mathrm{pair}} \]
\[ \mathcal{L}_{\mathrm{contrastive}}(h_{\mathrm{clean}}, h_{\mathrm{corrupt}}) \to S_{\mathrm{conf}}\ (\text{visual grounding}) \ \to\ L_{\mathrm{ground}} \]</div>
<p>Written out for a mini-batch:</p>
<div class="eq">\[ L = L_{\mathrm{cls}} + w_{\mathrm{pre}}L_{\mathrm{pre}} + w_{\mathrm{pair}}L_{\mathrm{pair}} + \lambda\,L_{\mathrm{ground}} \]
\[ L_{\mathrm{ground}} = \operatorname*{mean}_{m\in\text{modes}}\ \operatorname*{mean}_{i\in\text{correct}} \operatorname{softplus}\bigl(\text{margin} - (s_m(x^{\mathrm{clean}}_i) - s_m(x^{\mathrm{corrupt}}_i))\bigr),\qquad \text{margin}=0 \]
\[ L_{\mathrm{pair}} = \operatorname*{mean}_{i\in\text{wrong}} \operatorname{softplus}\bigl(-(s(x^{\mathrm{ref}}_i) - s(x^{\mathrm{pred}}_i))\bigr) \]</div>
<p>Here \(x^{\mathrm{clean}}\) is the real image with the question and the model's answer, \(x^{\mathrm{corrupt}}\) is the same question and answer text with a corrupted image, \(x^{\mathrm{ref}}\) appends the correct answer and \(x^{\mathrm{pred}}\) appends the model's own answer. The old loss is the case \(\lambda = 0\). The grounding term is applied only to answers the model got right: for those, the evidence is genuinely in the image, so removing the image should lower the score. For wrong answers there is nothing to ground, so they enter only through the correctness loss. The second line (the pair term) is our reading of the proposal's second contrastive term; an embedding-level contrastive loss would need a change to the head loss and was not tried.</p>
<h3>9.3 Corrupting the image: swap and noise</h3>
<p>The grounding term needs a corrupted image. Two were built, in this order.</p>
<ul>
<li><b>Swap.</b> Replace the image with a different example's image from the same split. Simple and strong, but it has a weakness: the corrupted pair is also a mismatched question and image, so a head can satisfy the term by detecting the mismatch, without judging how well the image supports the answer.</li>
<li><b>Noise.</b> Keep the right image and mix Gaussian noise into it, the forward step of a diffusion process (Ho et al. 2020): \(x' = \sqrt{\alpha}\,x + \sqrt{1-\alpha}\,\varepsilon\), \(\varepsilon\sim\mathcal{N}(0,I)\), with \(\alpha = 0.15\). The scene is still there but its detail is not, so there is no mismatch to detect. The strength was chosen from a smoke test of 30 examples per dataset, by how far the model's own answer probability falls.</li>
</ul>
<p>Mask (60% of a 16 by 16 grid blanked) and patch shuffle (a 4 by 4 grid permuted) were also built. They were extracted for the test splits and used only to grade heads on corruptions they never trained on. Everything was run through the same pipeline: a pooled sweep for each corruption, a re-grade of the swap heads under the other corruptions, one sweep per dataset alone for each corruption, and sequential training (Section 10). Details: reports <a href="@@l05@@">05</a> (swap) and <a href="@@l06@@">06</a> (noise).</p>
<h3>9.4 How grounding is measured</h3>
<p>For each correct test answer, the head scores the pair with the clean image and with the corrupted one. <b>Order accuracy</b> is the share of correct answers whose score falls under corruption (0.5 means no effect, 1.0 means it always falls). It is the number quoted in the results as "the score falls". The untrained answer-token probability can be graded the same way.</p>
<h3>9.5 Results: pooled mixture</h3>
@@pooled@@
<p>The old loss column is the same head in every version. Differences between the loss columns in AUROC are no more than a few thousandths, inside the seed spread. Grounding is graded under noise for the old loss and for the noise version, and under swap for the swap version, as that is the corruption each head trained against.</p>
<h3>9.6 What each term does (2 by 2 ablation, swap)</h3>
<p>The two contrastive terms were switched on and off independently, keeping everything else fixed.</p>
@@ablation@@
<div class="callout warn"><b class="t">Important</b>The grounding term is nearly free: it changes AUROC by at most 0.001 in either direction (unified +0.001, verifier -0.001) and raises the share of correct answers whose score falls under a corrupted image by 0.03 to 0.04. The pair term is not free: on its own it lowers AUROC by 0.01 to 0.02. The loss as first specified, with both terms, is therefore slightly worse than the old loss in ranking, and the grounding term alone is the version worth keeping.</div>
<h3>9.7 Does grounding learned on one corruption carry over?</h3>
<p>Each row is graded on the same test rows under four corruptions. The mask and shuffle columns are corruptions never trained on.</p>
@@transfer@@
@@figure@@
<p>Training with either corruption raises grounding on every other corruption compared with the old loss. For heads trained with swap the gain is about 0.03 to 0.04 on noise and mask and 0.015 to 0.024 on shuffle. For heads trained with noise the gain is largest on mask (0.05 to 0.06), which is expected because both remove detail instead of changing content. The gain is not specific to one corruption, although all four still alter the hidden state in ways a head can read.</p>
<h3>9.8 Each dataset alone</h3>
@@single@@
<p>On ChartQA, TextVQA and DocVQA the old-loss head already lowers its score on about 0.97 or more of correct answers under noise (all three head kinds), so there is little room and AUROC does not move. ScienceQA is where the term matters: the old loss lowers the score on 0.69 to 0.78 of correct answers under noise across the three head kinds, close to indifferent, and the noise loss raises that to 0.87 to 1.0 (the table shows the unified head: 0.72 to 0.94). ScienceQA questions can often be answered from the text alone, so a score that ignores the image is the expected outcome without the term.</p>
<div class="callout warn"><b class="t">How far to trust the grounding numbers</b>Order accuracy near 1.0 is partly easy. A noised image moves the hidden state a lot, so a head can learn to lower its score on noisy-looking inputs without checking whether the image supports the answer. The cross-corruption check lowers this worry but does not remove it. A stronger test is a corruption that removes only the evidence for the answer (for example, masking only the region the answer comes from), which was not run. The noise strength was set on a 30-example smoke test and only one strength was trained.</div>
"""
    return fill(text, l05=LINKS["05"], l06=LINKS["06"],
                pooled=table(["Head", "AUROC, old loss", "AUROC, new loss (swap)", "AUROC, new loss (noise)", "Falls under noise, old loss", "Falls under swap, new loss (swap)", "Falls under noise, new loss (noise)"], rows, left_cols=(0,)),
                ablation=table(["Head", "Terms on", "AUROC", "Share whose score falls (swap)", "AUROC change vs plain", "Grounding change vs plain"], ablation, left_cols=(0, 1)),
                transfer=transfer,
                figure=figure("fig-transfer", "Grounding on corruptions that were never trained on", pair_plot(plot_rows, 0.85, 1.0, "Share of correct answers whose score falls (old loss, new loss trained with noise)", "fig-transfer"), "Orange is the old loss and blue the noise-trained new loss. Both are graded on mask and shuffle."),
                single=table(["Dataset", "AUROC, old", "AUROC, new (swap)", "AUROC, new (noise)", "Falls under noise, old", "Falls under swap, new (swap)", "Falls under noise, new (noise)"], single_rows, left_cols=(0,)))


def section_sequential(r: Results) -> str:
    """Section 10: sequential training and forgetting."""
    variants = (("old_replay0", "Old loss, no replay"), ("old_replay0.1", "Old loss, 10% replay"), ("new_replay0", "New loss, no replay"), ("new_replay0.1", "New loss, 10% replay"))
    mode = {"pre": "main", "verifier": "main", "unified": "joint"}
    swap_rows, noise_rows = [], []
    for kind in KINDS:
        data_swap = r.continual.get(kind)
        data_noise = corr.load(f"mixture_noise_continual_{kind}")
        for key, label in variants:
            for data, rows, source in ((data_swap, swap_rows, "swap"), (data_noise, noise_rows, "noise")):
                if not data or key not in data["variants"]:
                    continue
                node = data["variants"][key]
                summary = node["summary"][mode[kind]]
                final = [entry["grounding_matrix"][mode[kind]][-1] for entry in node["orders"].values()]
                rows.append([KIND_NAME[kind], label, fmt(summary["final_avg"]["mean"]), fmt(summary["forgetting"]["mean"]), fmt(summary["bwt"]["mean"]), fmt(summary["retention"]["mean"]), fmt(sum(sum(row) / len(row) for row in final) / len(final))])
    maps = ""
    if "unified" in r.continual:
        for key, label in (("old_replay0", "Old loss"), ("new_replay0", "New loss")):
            entry = r.continual["unified"]["variants"][key]["orders"]["forward"]
            order = [NAMES.get(n, n) for n in entry["order"]]
            maps += figure(f"fig-seq-{key}", f"Unified head, {label.lower()}, forward order: AUROC on every dataset after every stage", heatmap(entry["matrix"]["joint"], [f"After {n}" for n in order], [NAMES.get(n, n) for n in r.datasets], f"fig-seq-{key}"), "Rows are stages, columns are the datasets scored. The cells below and to the left of the diagonal show datasets seen earlier.")
    iso = r.continual["unified"]["variants"]["old_replay0"]["isolated"]["joint"] if "unified" in r.continual else {}
    iso_rows = [[NAMES.get(k, k), pm((v["mean"], v["std"]), 3)] for k, v in iso.items()]
    text = r"""
<h2 id="seq">10. Stage 6: sequential training and forgetting</h2>
<p><b>The question.</b> The mixture trains on all datasets at once, which is the easy case. In practice a head may be trained on dataset A, then updated on B and C. Does it still score A well afterwards? This is catastrophic forgetting, and it decides whether the heads can be extended to new data without retraining from scratch.</p>
<h3>10.1 Protocol</h3>
<p>One head is trained through the four datasets in a fixed order, continuing from the previous weights, with early stopping on each stage's validation split. After every stage the head is scored on every dataset's test split, giving a stage-by-dataset matrix of AUROC. Three orders (forward, reverse, shuffled) and three seeds are averaged. The head configurations are the winners of the pooled sweeps, so the old and new loss are compared at the settings chosen on validation. Two variants of each loss were run: plain sequential training, and with a replay buffer, which mixes a fixed 10% random sample of every earlier dataset's training rows into each later stage.</p>
<p>Let \(M_{t,d}\) be the AUROC on dataset \(d\) after stage \(t\), \(T\) the number of stages, and \(\tau(d)\) the stage at which \(d\) is trained. The metrics are:</p>
<div class="eq">\[ \text{final average}=\tfrac{1}{D}\sum_d M_{T,d},\qquad \text{forgetting}=\tfrac{1}{D-1}\sum_{d\ne \text{last}}\Bigl(\max_{t\ge\tau(d)}M_{t,d}-M_{T,d}\Bigr) \]
\[ \text{backward transfer}=\tfrac{1}{D-1}\sum_{d\ne\text{last}}\bigl(M_{T,d}-M_{\tau(d),d}\bigr),\qquad \text{retention}=\tfrac{1}{D}\sum_d\bigl(M_{T,d}-M^{\text{alone}}_{d}\bigr) \]</div>
<p>Retention compares the final head with a head trained on that dataset alone. The last column of the tables is the grounding order accuracy after the final stage, averaged over datasets.</p>
<h3>10.2 Results with the swap loss</h3>
@@swap@@
<h3>10.3 Results with the noise loss</h3>
@@noise@@
@@maps@@
<h3>10.4 Heads trained on one dataset alone, for reference</h3>
@@iso@@
<div class="callout"><b class="t">Findings</b>Heads trained through all four datasets in sequence keep most of what they learned: forgetting is about 0.011 to 0.015 AUROC for the unified head, 0.013 to 0.025 for the pre-generation head, and the answer-conditioned verifier forgets the most (0.032 to 0.046 without replay). A 10% replay buffer cuts the verifier's forgetting to about 0.005 to 0.008 and changes little for the others. The new loss does not change forgetting much for the unified head, and the noise loss is slightly worse than the swap loss without replay on the pre-generation and verifier heads, with replay removing the difference. Grounding is not forgotten: after the last stage the share of correct answers whose score falls is higher with the new loss than with the old loss in every row.</div>
"""
    cols = ["Head", "Variant", "Final average AUROC", "Forgetting", "Backward transfer", "Retention", "Grounding after last stage"]
    return fill(text, swap=table(cols, swap_rows, left_cols=(0, 1)), noise=table(cols, noise_rows, left_cols=(0, 1)), maps=maps, iso=table(["Dataset", "Unified joint AUROC when trained alone"], iso_rows))


def section_heads() -> str:
    """Section 11: the trained heads in the repository."""
    root = REPO / "trained_heads" / "approach1" / "winners"
    rows = []
    total_files = total_configs = 0
    for tag, label in (("mixture", "Pooled mixture, swap"), ("mixture_noise", "Pooled mixture, noise"), ("chartqa_swap", "ChartQA alone, swap"), ("chartqa_noise", "ChartQA alone, noise"), ("textvqa_swap", "TextVQA alone, swap"), ("textvqa_noise", "TextVQA alone, noise"),
                       ("docvqa_swap", "DocVQA alone, swap"), ("docvqa_noise", "DocVQA alone, noise"), ("scienceqa_swap", "ScienceQA alone, swap"), ("scienceqa_noise", "ScienceQA alone, noise")):
        path = root / tag / "manifest.json"
        if not path.exists():
            continue
        manifest = load_json(path)
        files = sum(len(e["files"]) for e in manifest)
        total_files += files
        total_configs += len(manifest)
        rows.append([f"winners/{tag}", label, str(len(manifest)), str(files), f"{max(abs(e['difference']) for e in manifest):.4f}"])
    seq_files = len(list((REPO / "trained_heads" / "approach1" / "sequential").rglob("*.pt"))) if (REPO / "trained_heads").exists() else 0
    text = r"""
<h2 id="heads">11. The trained heads</h2>
<p>Every head that mattered for a result is saved in the repository under <code>trained_heads/approach1/</code>, with a loading guide in <code>trained_heads/README.md</code>. The sweeps kept weights only in memory, so the winning configurations were retrained with the same data, seeds and early stopping and checked against the sweep results.</p>
<ul>
<li><b>@@winners@@ sweep winners</b> (@@configs@@ configurations on three seeds): every winning configuration for the pooled mixture, the noise mixture, and each dataset alone with swap and with noise, for all head kinds and for the old loss, new loss, linear probes and the four ablation cells.</li>
<li><b>@@seq@@ sequential heads</b>: the final head of each sequential variant and order (seed 1), for the swap and noise mixtures.</li>
</ul>
@@table@@
<p>The last column is the largest gap between the three-seed mean test AUROC of the saved heads and the sweep's mean. The pooled and noise folders reproduce to under 0.001, and the largest gap anywhere is 0.005, about the seed spread. The Qwen3-VL-2B model itself is not included (it is never changed) and the cached features are regenerated with <code>extract_q_features.py</code>.</p>
<pre>from pathlib import Path
from experiments.approach1_latent_self_correction.export_heads import load_head, score
head, spec, temperatures = load_head(Path("trained_heads/approach1/winners/mixture_noise/unified_new_seed1.pt"))
probabilities = score(head, spec, temperatures, prompt_state, answer_state)   # {"joint": ..., "pre_generation": ...}</pre>
<p>These exported heads are the ones the correction experiments of Section 12 load, which shows they work as a component and not only as a table of results.</p>
"""
    return fill(text, winners=total_files, configs=total_configs, seq=seq_files, table=table(["Folder", "Trained on", "Configurations", "Files", "Largest gap to sweep mean"], rows, left_cols=(0, 1)))


def correction_tables() -> dict:
    """Load the correction experiment summaries and per-example records."""
    names = {
        "retry": "mixture_test_2b_prompted_retry_limit100.summary.json",
        "rerank": "mixture_test_2b_prompted_retry_reranked_limit100.summary.json",
        "best": "mixture_test_2b_best_of_five_limit100.summary.json",
        "deep": "mixture_test_2b_best_of_five_deep_limit100.summary.json",
    }
    data = {key: load_json(RESULTS / name) for key, name in names.items() if (RESULTS / name).exists()}
    deep_records = RESULTS / "mixture_test_2b_best_of_five_deep_limit100.jsonl"
    if deep_records.exists():
        data["deep_records"] = [json.loads(line) for line in deep_records.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = RESULTS / "mixture_test_2b_best_of_five_limit100.jsonl"
    if records.exists():
        data["records"] = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines() if line.strip()]
    return data


def auroc_of(labels, scores) -> float:
    """AUROC from ranks, ties given average rank."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    return (sum(rank for rank, (_, label) in zip(ranks, pairs) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def section_correction(data: dict) -> str:
    """Section 12: the correction experiments."""
    if not data:
        return '<h2 id="correction">12. Correction experiments</h2><p>The correction experiment results were not available when this report was built.</p>'
    order = [k for k, _ in DATASETS]

    def retry_table(summary, keys, labels):
        rows = []
        for key in keys:
            if key not in summary["policies"]:
                continue
            p = summary["policies"][key]
            pooled = p["pooled"]
            rows.append([labels[key], pct(pooled["final_accuracy"]), str(pooled["flagged"]), f"+{pooled['fixed_errors']}", f"-{pooled['broken_answers']}"] + [pct(p["per_dataset"][d]["final_accuracy"], 0) for d in order])
        return table(["Policy", "Accuracy", "Retried", "Fixed", "Broken"] + [n for _, n in DATASETS], rows, left_cols=(0,))

    retry_labels = {"baseline_no_retry": "No retry (greedy answer kept)", "retry_all": "Retry every answer", "head_threshold": "Retry if head score below 0.5", "head_lowest_fraction": "Retry the 20% with the lowest head score",
                    "token_lowest_fraction": "Retry the 20% with the lowest answer-token probability", "random_matched_fraction": "Retry a random 20% (control)"}
    rerank_labels = {**retry_labels, "head_rerank": "Head reranking: keep the higher-scored of original and retry", "head_gated_margin_0.00": "Gated: retry only if original below 0.5 and retry scores higher",
                     "head_gated_margin_0.05": "Gated, margin 0.05", "head_gated_margin_0.10": "Gated, margin 0.10", "head_gated_margin_0.20": "Gated, margin 0.20", "oracle_upper_bound": "Oracle: use the retry exactly when it fixes an error"}
    best_labels = {"greedy": "Greedy answer", "first_sample": "First of five samples", "majority_vote": "Majority vote of five", "token_best": "Highest answer-token probability", "head_best": "Highest head score", "oracle": "Oracle: any correct candidate"}
    best_rows, chart_rows = [], []
    for key, label in best_labels.items():
        p = data["best"]["policies"][key] if "policies" in data["best"] else data["best"][key]
        pooled = p["pooled"]
        low, high = wilson(pooled["correct"], pooled["examples"])
        best_rows.append([label, pct(pooled["accuracy"]), f"{pct(low, 0)} to {pct(high, 0)}"] + [pct(p["per_dataset"][d]["accuracy"], 0) for d in order])
        chart_rows.append((label, pooled["accuracy"], "c-uj" if key == "head_best" else "c-tok", None))
    deep_rows = []
    deep_policies = data["deep"]["policies"] if "deep" in data else {}
    deep_labels = {"keep_best_of_five": "Keep the head-selected candidate", "deep_replace_all": "Replace with the deeper answer, always", "deep_head_rerank": "Keep the higher head score of the two", "deep_margin_0.00": "Deeper answer if its head score is higher",
                   "deep_margin_0.05": "Deeper answer if at least 0.05 higher", "deep_margin_0.10": "Deeper answer if at least 0.10 higher", "deep_margin_0.20": "Deeper answer if at least 0.20 higher", "deep_oracle": "Oracle: deeper answer exactly when it fixes the candidate"}
    for key, label in deep_labels.items():
        if key in deep_policies:
            p = deep_policies[key]
            pooled = p["pooled"]
            deep_rows.append([label, pct(pooled.get("final_accuracy", pooled.get("accuracy", float("nan")))), f"+{pooled.get('fixed_errors', 0)}", f"-{pooled.get('broken_answers', 0)}"] + [pct(p["per_dataset"][d].get("final_accuracy", p["per_dataset"][d].get("accuracy", float("nan"))), 0) for d in order])
    records = data.get("records", [])
    stats = ""
    fixed = broken = 0
    cand_auc = tok_auc = None
    if records:
        for rec in records:
            head = rec["candidates"][rec["selected_indices"]["head_best"]]["correct"]
            greedy = rec["greedy"]["correct"]
            fixed += int(head and not greedy)
            broken += int(greedy and not head)
        labels = [int(c["correct"]) for rec in records for c in rec["candidates"]]
        cand_auc = auroc_of(labels, [c["head_confidence"] for rec in records for c in rec["candidates"]])
        tok_auc = auroc_of(labels, [c["token_confidence"] for rec in records for c in rec["candidates"]])
        stats = f"""<div class="callout"><b class="t">How strong is the evidence?</b>Each study has 100 examples. At that size, 81% accuracy has a 95% interval of about {pct(wilson(81, 100)[0], 0)} to {pct(wilson(81, 100)[1], 0)} and 78% has {pct(wilson(78, 100)[0], 0)} to {pct(wilson(78, 100)[1], 0)}, so the intervals overlap heavily. Head selection changed the outcome of {fixed + broken} questions against greedy: it fixed {fixed} and broke {broken}. A paired sign test on {fixed} against {broken} gives a two-sided p-value of about 0.38. The direction is encouraging, but the experiment is too small to claim the improvement is real. A run on the full test split is the next step. The candidate-level check is cleaner: over the {len(labels)} sampled answers, the head's score has AUROC {cand_auc:.3f} for telling correct from wrong candidates, against {tok_auc:.3f} for the answer-token probability.</div>"""
    deep_recs = data.get("deep_records", [])
    cap_n = long_n = cap_broken = echo_n = echo_broken = fmt_broken = 0
    echo_where: dict = {}
    if deep_recs:
        for rec in deep_recs:
            tokens = len(rec["deep_answer"]["token_probabilities"])
            before, after = rec["selected_candidate"]["correct"], rec["deep_answer"]["correct"]
            broke = before and not after
            echoed = rec["deep_answer"]["answer"].startswith("A candidate answer was")
            capped = tokens >= 64
            long_n += int(len(rec["deep_answer"]["answer"]) > 60)
            cap_n += int(capped)
            cap_broken += int(capped and broke)
            echo_n += int(echoed and not capped)
            if echoed and not capped:
                echo_where[rec["dataset"]] = echo_where.get(rec["dataset"], 0) + 1
            echo_broken += int(echoed and broke)
            fmt_broken += int((capped or echoed) and broke)
    genuine = sum(1 for rec in deep_recs if rec["selected_candidate"]["correct"] and not rec["deep_answer"]["correct"]) - fmt_broken
    cases = ""
    if records:
        rows = []
        for rec in records:
            chosen = rec["candidates"][rec["selected_indices"]["head_best"]]
            greedy = rec["greedy"]
            if chosen["correct"] == greedy["correct"]:
                continue
            question = " ".join(str(rec["question"]).split())
            question = question if len(question) <= 150 else question[:147] + "..."
            rows.append([f"{rec['dataset']}:{rec['index']}", esc(question), esc(rec["reference_answer"]),
                         f"{esc(greedy['answer'])} ({'correct' if greedy['correct'] else 'wrong'})",
                         f"{esc(chosen['answer'])} (head {chosen['head_confidence']:.3f}, {'correct' if chosen['correct'] else 'wrong'})"])
        cases = ("<h4>Cases where head selection changed the outcome</h4><p class=\"small\">Every example where the head-selected candidate was right and greedy was wrong, or the reverse. Long ScienceQA contexts are shortened.</p>"
                 + table(["Example", "Question", "Reference", "Greedy answer", "Head-selected candidate"], rows, left_cols=(0, 1, 2, 3, 4)))
    text = r"""
<h2 id="correction">12. Stage 8: using the head to correct answers</h2>
<p>The previous sections built the gate. These experiments test whether acting on it helps. Sections 12.1 to 12.5 are the pilot, run on 100 examples; Section 12.6 repeats everything on the full test split and supersedes the pilot numbers. All of them use the frozen Qwen3-VL-2B and the exported pooled-mixture unified head (<code>trained_heads/approach1/winners/mixture/unified_new_seed1.pt</code>), on a balanced study set of 100 examples, 25 from each dataset, of the pooled test split. These experiments were pushed to the <code>feature/latent-self-correction</code> branch (commit <code>e6aaa80</code>) and are not on <code>main</code> yet. The code is in <code>prompted_retry.py</code>, <code>online_prompted_retry.py</code>, <code>candidate_reranking.py</code> and <code>deepen_selected_candidate.py</code>, run by <code>run_prompted_retry.sbatch</code>, <code>run_candidate_reranking.sbatch</code> and <code>run_deepen_selected.sbatch</code>, with sampling options added to <code>models/run_inference.py</code> and tests in <code>tests/test_prompted_retry.py</code>. The full report is <a href="@@l07@@">07</a>. Accuracy is the dataset's own metric converted to the binary correctness definition used to train the head.</p>
<div class="callout warn"><b class="t">About "chain of thought"</b>The deeper pass (Section 12.4) is the experiment described as chain of thought: after a candidate is chosen, the model is told to <em>think more deeply</em>, re-examine the visual evidence and correct the candidate. It ran on the same 100 examples with a limit of 64 new tokens. The prompt still ends with <em>respond with only the final answer, without explanation</em>, and most of the time the model complied with a short answer. It did not always. In @@capn@@ cases (all ChartQA) it wrote out reasoning anyway, such as a list of the chart values or a subtraction, and ran into the 64-token limit before reaching a final answer. In @@echon@@ further cases (@@where@@) it repeated the prompt text instead of answering. So the run mixes direct answers, a few cut-off chains of thought and a few format failures. It is not a controlled chain-of-thought experiment: reasoning was not requested, was not given enough tokens, and no final answer was extracted from it. A proper run (ask for reasoning, allow enough tokens, extract the final answer) has not been done.</div>
<h3>12.1 Prompted retry</h3>
<p><b>Idea.</b> The model answers once. A second prompt shows it the previous answer and asks it to inspect the visual evidence again. The head decides <em>when</em> to retry, so the retry's cost and its risk of overturning correct answers are spent only where the gate thinks the first answer is weak. Each policy is evaluated on the same retry outputs, so there is no generation noise between rows. A random 20% control has the same compute as the head-targeted policies.</p>
<pre>Your previous answer was: {answer}
Try the question again. Carefully inspect the visual evidence, including labels, values, and spatial relationships. Do not assume the previous answer is correct.
Respond with only the final answer, without explanation.</pre>
@@retry@@
<p>This first run began at 81% for the greedy answer. Retrying every answer lowered accuracy to 78%, and head-based targeting also reached 78%, while the random control reached 82%. In this run a single prompted retry is unstable: it fixes some errors but overturns more correct answers, and the head's flagging did not beat random choice.</p>
<h3>12.2 Scored retry and reranking</h3>
<p><b>Idea.</b> Instead of trusting the retry, score both the original and the retry with the same head and keep the one with the higher score. Gated variants also require the original to score below 0.5 and optionally require a minimum improvement, so that the retry replaces the original only with good evidence.</p>
@@rerank@@
<p>This run was generated independently and its greedy baseline is 79%. Here the head-targeted retries (below 0.5, or the lowest 20%) reached 80% with 3 fixes and 2 breaks, above the random control (78%) and the baseline (79%), and the gated reranking also reached 80% (2 fixes, 1 break). The oracle is 83%. So in this run the head-guided retry is one point better than doing nothing, and in the first run it was three points worse than doing nothing. The two runs differ because of generation and GPU nondeterminism, and the effect in both is within a few questions out of 100, so the honest reading is that a prompted retry does not clearly help, with or without the head. Comparisons are only meaningful within a table.</p>
<h3>12.3 Best-of-five candidate selection</h3>
<p><b>Idea.</b> Separate creating candidates from choosing between them. Five answers are sampled at temperature 0.9 and top-p 0.95. The head scores every candidate from its answer-state \(h_a\), and the highest score is selected: \(\hat a = \arg\max_k\, p_\theta(\text{correct} \mid h_q, h_{a_k})\). The comparison methods are the first sample, a majority vote over normalized answer text, the sample with the highest answer-token probability, the greedy baseline, and an oracle that picks a correct candidate whenever one exists (an unattainable upper bound).</p>
@@best@@
@@bestchart@@
@@stats@@
@@cases@@
<p>Candidate sets contained @@unique@@ unique answers on average and all five samples agreed on @@unanimous@@ of examples. Head selection improved greedy accuracy by three points and captured three of the seven points between greedy and the oracle. The head selector (81%) scored one point above the majority vote (80%) and three above the answer-token probability (78%). These are small margins on 100 examples. Note that the first sample alone (80%) is already close to the head's choice, so most of the gain over greedy may come from sampling and not from selection; only the larger run can separate the two.</p>
<h3>12.4 A deeper pass after selection</h3>
<p><b>Idea.</b> After the head picks a candidate, the model sees that candidate and is asked to think more deeply and correct it. The head then decides whether the deeper answer replaces the candidate.</p>
@@deep@@
<p>Replacing every selected candidate with the deeper answer lowered accuracy from 81% to 73% (2 fixed, 10 broken). Every head-based replacement rule left the answers unchanged: the head never scored the deeper answer high enough to replace the selected candidate. The oracle for this step is 83%, so there is only a small upper bound to gain even with a perfect rule.</p>
<p>Is the drop only a format problem? Partly. Of the 10 answers the deeper pass broke, @@fmtb@@ are format failures: @@capb@@ was cut off at the 64-token limit and @@echob@@ repeated the prompt instead of answering. The other @@genuine@@ are real changes of a correct answer to a wrong one, against 2 real fixes. So fixing the format would shrink the loss but would not turn it into a gain on this sample, and a controlled run with a larger token budget and answer extraction is needed before concluding anything about chain of thought.</p>
<h3>12.5 What the pilot showed</h3>
<div class="callout"><b class="t">Findings</b>(1) The trained head works as a component: loaded from the repository and used to score freshly generated answers, it gives a candidate-level AUROC of @@cand@@ against @@tok@@ for the answer-token probability. (2) Asking the same model to retry or think more deeply is unreliable. It fixes some errors and breaks others. With this head as the gate the net effect ranged from -3 to +1 points between runs and policies, and the deeper pass was clearly harmful when applied to everything (-8 points). (3) Generating several candidates and letting the head choose looked promising in the pilot (81% against 78% greedy, 85% oracle). (4) All of this was on 100 examples, so none of the differences was statistically established, and the full-split rerun in Section 12.6 shows the best-of-five gain was mostly noise.</div>
"""
    unique = data["best"]["candidate_diversity"]["mean_unique_answers"] if "best" in data else float("nan")
    unanimous = data["best"]["candidate_diversity"]["unanimous_fraction"] if "best" in data else float("nan")
    return fill(text, where=", ".join(f"{n} in {NAMES.get(d, d)}" for d, n in sorted(echo_where.items(), key=lambda kv: -kv[1])), capn=cap_n, capb=cap_broken, echon=echo_n, echob=echo_broken, fmtb=fmt_broken, genuine=genuine, l07=LINKS["07"], retry=retry_table(data["retry"], list(retry_labels), retry_labels) if "retry" in data else "", rerank=retry_table(data["rerank"], list(rerank_labels), rerank_labels) if "rerank" in data else "",
                best=table(["Selection method", "Accuracy", "95% interval"] + [n for _, n in DATASETS], best_rows, left_cols=(0,)),
                bestchart=figure("fig-best", "Best-of-five selection on 100 examples", bar_chart(chart_rows, 0.7, 0.9, "Pooled accuracy (axis starts at 0.7)", "fig-best", lambda v: f"{100 * v:.0f}%"), "Blue is the head selector. With 100 examples, differences of a few points are within noise."),
                stats=stats, cases=cases, unique=unique, unanimous=pct(unanimous, 0), deep=table(["Policy", "Accuracy", "Fixed", "Broken"] + [n for _, n in DATASETS], deep_rows, left_cols=(0,)) if deep_rows else "",
                cand=f"{cand_auc:.3f}" if cand_auc else "n/a", tok=f"{tok_auc:.3f}" if tok_auc else "n/a")



def full_correction_data() -> dict:
    """Load the merged full-test-split correction records, if the runs have finished."""
    names = {"retry": "mixture_test_2b_prompted_retry_reranked_full.jsonl", "best5": "mixture_test_2b_best_of_five_full.jsonl", "deep": "mixture_test_2b_best_of_five_deep_full.jsonl"}
    keys = {"retry": "baseline_correct", "best5": "candidates", "deep": "deep_answer"}
    data = {}
    for key, name in names.items():
        path = RESULTS / name
        if not path.exists():
            return {}
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        data[key] = [r for r in rows if keys[key] in r]
    return data


def signed(value: float, digits: int = 1) -> str:
    """A difference in accuracy as signed percentage points."""
    return f"{100 * value:+.{digits}f}"


def p_text(p: float) -> str:
    """A p-value for a table."""
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def section_correction_full(full: dict) -> str:
    """Section 12.6: every correction experiment on the full test split, with intervals and paired tests."""
    if not full:
        return ""
    from experiments.approach1_latent_self_correction import correction_stats as st
    retry, best5, deep = full["retry"], full["best5"], full["deep"]
    n = len(best5)
    order = [k for k, _ in DATASETS]

    def policy_table(headers, rows):
        return table(headers, rows, left_cols=(0,))

    greedy = st.best5_outcomes(best5, "greedy")
    labels = {"greedy": "Greedy answer", "first_sample": "First of five samples", "majority_vote": "Majority vote of five", "token_best": "Highest answer-token probability", "head_best": "Highest head score", "oracle": "Oracle: any correct candidate"}
    rows5, stats5 = [], {}
    for policy, label in labels.items():
        outcome = st.best5_outcomes(best5, policy)
        row = st.policy_row(greedy, outcome)
        stats5[policy] = row
        rows5.append([label, pct(row["accuracy"], 2), f"{pct(row['accuracy_low'], 1)} to {pct(row['accuracy_high'], 1)}", "" if policy == "greedy" else f"{signed(row['difference'], 2)} ({signed(row['ci_low'], 2)} to {signed(row['ci_high'], 2)})",
                      "" if policy == "greedy" else f"+{row['fixed']} / -{row['broken']}", "" if policy == "greedy" else p_text(row["p_value"])])
    per5 = []
    for dataset in order:
        sub = [r for r in best5 if r["dataset"] == dataset]
        g, h = st.best5_outcomes(sub, "greedy"), st.best5_outcomes(sub, "head_best")
        d = st.paired_difference(g, h)
        per5.append([dict(DATASETS)[dataset], f"{len(sub):,}", pct(sum(g) / len(g), 1), pct(sum(h) / len(h), 1), pct(sum(st.best5_outcomes(sub, "oracle")) / len(sub), 1), f"{signed(d['difference'], 2)}", f"+{d['fixed']} / -{d['broken']}", p_text(d["p_value"])])
    labels_c = [int(c["correct"]) for r in best5 for c in r["candidates"]]
    cand_head = st.auroc(labels_c, [c["head_confidence"] for r in best5 for c in r["candidates"]])
    cand_tok = st.auroc(labels_c, [c["token_confidence"] for r in best5 for c in r["candidates"]])
    head_vs_first = st.paired_difference(st.best5_outcomes(best5, "first_sample"), st.best5_outcomes(best5, "head_best"))
    head_vs_major = st.paired_difference(st.best5_outcomes(best5, "majority_vote"), st.best5_outcomes(best5, "head_best"))
    diversity = sum(len({norm(c["answer"]) for c in r["candidates"]}) for r in best5) / n

    retry_labels = {"baseline_no_retry": "No retry (greedy answer kept)", "retry_all": "Retry every answer", "head_threshold": "Retry if head score below 0.5", "head_lowest_fraction": "Retry the 20% with the lowest head score",
                    "token_lowest_fraction": "Retry the 20% with the lowest answer-token probability", "random_matched_fraction": "Retry a random 20% (control)", "head_rerank": "Head reranking: keep the higher-scored of original and retry",
                    "head_gated_margin_0.00": "Gated: retry only if original below 0.5 and retry scores higher", "head_gated_margin_0.05": "Gated, margin 0.05", "head_gated_margin_0.10": "Gated, margin 0.10", "head_gated_margin_0.20": "Gated, margin 0.20",
                    "oracle_upper_bound": "Oracle: use the retry exactly when it fixes an error"}
    rows_r, stats_r = [], {}
    for policy, label in retry_labels.items():
        base, fin = st.retry_outcomes(retry, st.RETRY_POLICIES[policy])
        row = st.policy_row(base, fin)
        stats_r[policy] = row
        retried = sum(1 for r in retry if r.get(st.RETRY_POLICIES[policy])) if policy != "baseline_no_retry" else 0
        rows_r.append([label, f"{retried:,}", pct(row["accuracy"], 2), "" if policy == "baseline_no_retry" else f"{signed(row['difference'], 2)} ({signed(row['ci_low'], 2)} to {signed(row['ci_high'], 2)})",
                       "" if policy == "baseline_no_retry" else f"+{row['fixed']} / -{row['broken']}", "" if policy == "baseline_no_retry" else p_text(row["p_value"])])
    per_r = []
    for dataset in order:
        sub = [r for r in retry if r["dataset"] == dataset]
        cells = [dict(DATASETS)[dataset], pct(sum(r["baseline_correct"] for r in sub) / len(sub), 1)]
        for policy in ("retry_all", "head_lowest_fraction", "random_matched_fraction", "head_gated_margin_0.00"):
            base, fin = st.retry_outcomes(sub, st.RETRY_POLICIES[policy])
            cells.append(f"{signed(sum(fin) / len(fin) - sum(base) / len(base), 2)}")
        per_r.append(cells)

    deep_labels = {"keep_best_of_five": "Keep the head-selected candidate", "deep_replace_all": "Replace with the deeper answer, always", "deep_head_rerank": "Keep the higher head score of the two", "deep_margin_0.00": "Deeper answer if its head score is higher",
                   "deep_margin_0.05": "Deeper answer if at least 0.05 higher", "deep_margin_0.10": "Deeper answer if at least 0.10 higher", "deep_margin_0.20": "Deeper answer if at least 0.20 higher", "deep_oracle": "Oracle: deeper answer exactly when it fixes the candidate"}
    rows_d, stats_d = [], {}
    for policy, label in deep_labels.items():
        base, fin = st.deep_outcomes(deep, st.DEEP_POLICIES[policy])
        row = st.policy_row(base, fin)
        stats_d[policy] = row
        rows_d.append([label, pct(row["accuracy"], 2), "" if policy == "keep_best_of_five" else f"{signed(row['difference'], 2)} ({signed(row['ci_low'], 2)} to {signed(row['ci_high'], 2)})",
                       "" if policy == "keep_best_of_five" else f"+{row['fixed']} / -{row['broken']}", "" if policy == "keep_best_of_five" else p_text(row["p_value"])])
    fail = st.format_failures(deep)
    all_broken = stats_d["deep_replace_all"]["broken"]
    format_broken = fail["broken_capped"] + fail["broken_echoed"]

    pilot = correction_tables()
    pilot_rows = []
    if pilot:
        pilot_rows = [
            ["Best-of-five: head selection minus greedy", signed(pilot["best"]["policies"]["head_best"]["pooled"]["accuracy"] - pilot["best"]["policies"]["greedy"]["pooled"]["accuracy"], 1) + " points (n=100)", f"{signed(stats5['head_best']['difference'], 2)} points (n={n:,})"],
            ["Best-of-five: oracle minus greedy", signed(pilot["best"]["policies"]["oracle"]["pooled"]["accuracy"] - pilot["best"]["policies"]["greedy"]["pooled"]["accuracy"], 1) + " points", f"{signed(stats5['oracle']['difference'], 2)} points"],
            ["Prompted retry of everything", signed(pilot["retry"]["policies"]["retry_all"]["pooled"]["final_accuracy"] - pilot["retry"]["policies"]["baseline_no_retry"]["pooled"]["final_accuracy"], 1) + " points", f"{signed(stats_r['retry_all']['difference'], 2)} points"],
            ["Retry the 20% with the lowest head score", signed(pilot["retry"]["policies"]["head_lowest_fraction"]["pooled"]["final_accuracy"] - pilot["retry"]["policies"]["baseline_no_retry"]["pooled"]["final_accuracy"], 1) + " points", f"{signed(stats_r['head_lowest_fraction']['difference'], 2)} points"],
            ["Retry a random 20% (control)", signed(pilot["retry"]["policies"]["random_matched_fraction"]["pooled"]["final_accuracy"] - pilot["retry"]["policies"]["baseline_no_retry"]["pooled"]["final_accuracy"], 1) + " points", f"{signed(stats_r['random_matched_fraction']['difference'], 2)} points"],
            ["Gated reranking (margin 0)", signed(pilot["rerank"]["policies"]["head_gated_margin_0.00"]["pooled"]["final_accuracy"] - pilot["rerank"]["policies"]["baseline_no_retry"]["pooled"]["final_accuracy"], 1) + " points", f"{signed(stats_r['head_gated_margin_0.00']['difference'], 2)} points"],
            ["Deeper pass applied to everything", signed(pilot["deep"]["policies"]["deep_replace_all"]["pooled"].get("final_accuracy", 0) - pilot["deep"]["policies"]["keep_best_of_five"]["pooled"].get("final_accuracy", 0), 1) + " points", f"{signed(stats_d['deep_replace_all']['difference'], 2)} points"],
        ]
    head = stats5["head_best"]
    gate = stats_r["head_gated_margin_0.00"]
    text = r"""
<h3>12.6 The full test split</h3>
<p>The 100-example runs were a pilot. Every experiment above was rerun on the <b>full test splits of all four datasets: @@n@@ questions</b> (ChartQA 2,500, TextVQA 2,500, DocVQA 2,500, ScienceQA 2,017), with the same prompts, sampling settings (temperature 0.9, top-p 0.95, five candidates), the same exported unified head, and the same 64-token limit. Each experiment ran as eight shards on separate GPUs and the shards were merged afterwards, so the policies that rank the whole set (the lowest 20% by head score, the random control) are computed once over all questions. There were no failed examples. Differences from a baseline are paired (the same questions under both policies) and come with a bootstrap 95% interval and an exact sign test (McNemar) on the questions where the two policies disagree. Accuracy differences are in percentage points.</p>
<div class="callout"><b class="t">Result in one paragraph</b>On the full test split the 100-example optimism does not hold. Choosing among five samples with the head improves on greedy generation by only @@headdiff@@ points (95% interval @@headlo@@ to @@headhi@@, p = @@headp@@), which is not distinguishable from zero, against +3 points in the pilot. Prompted retry lowers accuracy unless it is tightly gated, where it gains at most +0.2 points. The deeper pass lowers accuracy by @@deepdiff@@ points when applied to everything. The oracles show real headroom (+@@oracle5@@ points for best-of-five, +@@oracleretry@@ for retry), so the head's scores rank candidates well (AUROC @@candhead@@ against @@candtok@@ for the answer-token probability over @@ncand@@ sampled answers) but that ranking is not yet turned into accuracy.</div>
<h4>Best-of-five candidate selection</h4>
@@t5@@
<p>Differences are against the greedy answer; "fixed / broken" counts questions the policy got right where greedy was wrong and the reverse. Candidate sets contained @@diversity@@ distinct answers on average. Head selection is @@vsfirst@@ points better than taking the first sample (p &lt; 0.001), so sampling alone does not help (the first sample is worse than greedy) and selection recovers that loss, but it does not beat greedy. The head is @@vsmajor@@ points above majority vote, which is also not significant.</p>
@@t5d@@
<p>Only TextVQA shows a gain larger than noise (+1.0 points, p = 0.04, one of four datasets, so not corrected for multiple comparisons). On ChartQA, where greedy is weakest, head selection is slightly worse than greedy.</p>
<h4>Prompted retry and reranking</h4>
@@tr@@
<p>Differences are against keeping the greedy answer. Retrying everything loses @@retryall@@ points: the retry fixes @@rfix@@ questions and breaks @@rbrk@@. Sending only the lowest 20% by head score to the retry loses less (@@headfrac@@ points) than the random control (@@randfrac@@) and the answer-token probability (@@tokfrac@@), so the head does point the retry at the right questions more often than chance, but the retry itself overturns too many correct answers for that to pay off. The gated policies (retry only if the original scores below 0.5 and the retry scores higher, optionally by a margin) are the only ones that help, by @@gate@@ points for margin 0 and about the same for 0.05, with intervals that touch zero.</p>
@@tr2@@
<p>Change in accuracy by dataset, in points. ScienceQA is where retrying everything hurts most (-7.0), and where gating helps most (+1.2).</p>
<h4>Deeper pass after best-of-five</h4>
@@td@@
<p>Applied to every selected answer, the deeper pass lowers accuracy from @@keep@@ to @@deepacc@@: it fixes @@dfix@@ questions and breaks @@dbrk@@. A large share of that is format failure and not wrong reasoning: @@capped@@ of the @@n@@ deeper answers hit the 64-token limit while writing out reasoning, and @@echoed@@ repeated the prompt text. Those two kinds account for @@fmtbroken@@ of the @@dbrk@@ breaks (@@fmtshare@@); the other @@genuine@@ are real changes of a correct answer to a wrong one. So the pass is clearly harmful as designed, and part of the harm could be removed by a larger token budget and answer extraction, but the genuine breaks alone (@@genuine@@) still outnumber the fixes (@@dfix@@). Letting the head decide gives at most +0.1 points.</p>
@@pilot@@
<div class="callout warn"><b class="t">What this changes in the conclusions</b>(1) Do not quote the pilot's "81% against 78%": on 9,517 questions head-selected best-of-five is +@@headdiff1@@ points over greedy and not statistically significant. (2) The detection work stands: the head ranks sampled answers far better than the answer-token probability (AUROC @@candhead@@ against @@candtok@@). (3) Turning that ranking into accuracy needs a stronger lever than choosing among five samples or asking the model to look again; the oracle gaps (+@@oracle5@@ points for best-of-five, +@@oracleretry@@ for retry) say the room exists. (4) The retry and deeper pass should be rerun with a controlled format (enough tokens, final-answer extraction) before drawing conclusions about chain of thought.</div>
"""
    return fill(text, n=f"{n:,}", headdiff=signed(head["difference"], 2), headdiff1=signed(head["difference"], 1).lstrip("+"), headlo=signed(head["ci_low"], 2), headhi=signed(head["ci_high"], 2), headp=p_text(head["p_value"]),
                deepdiff=signed(stats_d["deep_replace_all"]["difference"], 1).lstrip("-"), oracle5=signed(stats5["oracle"]["difference"], 1).lstrip("+"), oracleretry=signed(stats_r["oracle_upper_bound"]["difference"], 1).lstrip("+"),
                candhead=f"{cand_head:.3f}", candtok=f"{cand_tok:.3f}", ncand=f"{len(labels_c):,}", diversity=f"{diversity:.2f}", vsfirst=signed(head_vs_first["difference"], 1).lstrip("+"), vsmajor=signed(head_vs_major["difference"], 1).lstrip("+"),
                retryall=signed(-stats_r["retry_all"]["difference"], 1).lstrip("+"), rfix=stats_r["retry_all"]["fixed"], rbrk=stats_r["retry_all"]["broken"], headfrac=signed(-stats_r["head_lowest_fraction"]["difference"], 2).lstrip("+"),
                randfrac=signed(-stats_r["random_matched_fraction"]["difference"], 2).lstrip("+"), tokfrac=signed(-stats_r["token_lowest_fraction"]["difference"], 2).lstrip("+"), gate=signed(gate["difference"], 2),
                keep=pct(stats_d["keep_best_of_five"]["accuracy"], 1), deepacc=pct(stats_d["deep_replace_all"]["accuracy"], 1), dfix=stats_d["deep_replace_all"]["fixed"], dbrk=all_broken, capped=f"{fail['capped']:,}", echoed=f"{fail['echoed']:,}",
                fmtbroken=format_broken, fmtshare=pct(format_broken / all_broken, 0), genuine=all_broken - format_broken,
                t5=policy_table(["Selection method", "Accuracy", "95% interval", "Difference from greedy (95% interval)", "Fixed / broken", "p"], rows5),
                t5d=policy_table(["Dataset", "Questions", "Greedy", "Head-selected", "Oracle", "Difference", "Fixed / broken", "p"], per5),
                tr=policy_table(["Policy", "Questions retried", "Accuracy", "Difference from no retry (95% interval)", "Fixed / broken", "p"], rows_r),
                tr2=policy_table(["Dataset", "Greedy accuracy", "Retry all", "Lowest 20% by head", "Random 20%", "Gated, margin 0"], per_r),
                td=policy_table(["Policy", "Accuracy", "Difference from the selected candidate (95% interval)", "Fixed / broken", "p"], rows_d),
                pilot=("<h4>Pilot against full split</h4>" + table(["Comparison", "Pilot, 100 examples", f"Full split, {n:,} questions"], pilot_rows, left_cols=(0,)) + "<p>The pilot overstated every effect that looked like a gain. With 100 examples, one question is one point, so differences of two to four points are a handful of questions; the full split shows they were noise.</p>") if pilot_rows else "")


def section_engineering() -> str:
    """Section 13: reproducibility and engineering."""
    return r"""
<h2 id="engineering">13. Engineering and reproducibility</h2>
<ul>
<li><b>Everything runs on the GT ICE cluster</b> with one command per stage. <code>run_qhead_pipeline.sh</code> runs a dataset end to end (splits, baseline, features, sweeps, analysis), <code>run_mixture_pipeline.sh</code> the mixture and swap loss, <code>run_continual_pipeline.sh</code> sequential training, <code>run_corruption_pipeline.sh</code> the noise loss and its cross-corruption checks, and <code>run_export_pipeline.sh</code> saves the trained heads. Each submits Slurm jobs with dependencies.</li>
<li><b>Scripts are checked before long runs.</b> <code>check_scripts.sh</code> validates shell syntax, that every <code>python -m</code> module resolves, and (on the cluster) that the scheduler accepts the resource request. Time limits are realistic on purpose, because the cluster caps the total GPU minutes a user can request.</li>
<li><b>36 offline tests</b> cover the scoring rules, the splits, the unified head, the corruptions, the old and new loss (including a check that the old loss is reproduced exactly when the new terms are off), the grounding metrics, the mixture builder, sequential training, and that the reports build from the real result schema. They need no GPU or network.</li>
<li><b>Feature caching</b> is what makes the experiments affordable. After one pass of the language model, a pooled sweep (36 or 72 head configurations, three seeds each, on 50,517 training rows) took 3 to 12 minutes per head kind on a single GPU, and a sequential-training run took 1.5 to 3.5 minutes.</li>
<li><b>Layout.</b> <code>models/</code> (model loading and inference), <code>data/</code> (dataset acquisition and scoring inputs), <code>experiments/</code> (training, analysis, reports), <code>slurm/</code>, <code>results/approach1/</code> (numbered reports and data), <code>trained_heads/</code>, <code>docs/</code>, <code>tests/</code>.</li>
</ul>
<p>The design note <code>docs/approach1_unified_q_head.md</code> records the dataset choices and the reasons other candidates were rejected.</p>
"""


def section_meaning(r: Results, swap: dict, noise: dict) -> str:
    """Section 14: what it all means, limits, next steps, and talking points."""
    tok = r.baseline("pooled")["token_confidence"]["correctness_auroc"]
    uni = r.metric("unified", "old", "joint", "correctness_auroc")[0]
    g_old = corr.grounding_against(noise["unified"], "old", "unified", "noise")
    g_new = corr.grounding_against(noise["unified"], "new", "unified", "noise")
    text = r"""
<h2 id="meaning">14. What it all means</h2>
<h3>14.1 What is established</h3>
<ul>
<li><b>The detection half works.</b> A small head on frozen hidden states predicts whether an answer is correct far better than the model's own token probability, on four different kinds of input (AUROC @@uni@@ against @@tok@@ pooled), and the ChartQA result reproduces end to end.</li>
<li><b>The signal is cheap.</b> A linear probe with about 2,000 parameters already captures most of the gain; the largest head has 2.1 million parameters, about 0.1% of the language model.</li>
<li><b>One head is enough.</b> The unified head does the job of two with no measurable loss, and one head trained on a pool of datasets stays within 0.015 AUROC of heads trained on one dataset.</li>
<li><b>Heads can be extended.</b> Training on one dataset after another loses 0.011 to 0.046 AUROC depending on the head and the loss (the verifier forgets most), and a 10% replay buffer removes most of that for the verifier.</li>
<li><b>A grounding term is cheap.</b> It makes the score react to a corrupted image (the share of correct answers whose score falls goes from @@gold@@ to @@gnew@@ under noise for the unified head) without costing ranking quality, and the effect carries over to corruptions it was not trained on. The pair term, applied alone, costs AUROC.</li>
</ul>
<h3>14.2 What is not established</h3>
<ul>
<li><b>Correction.</b> Nothing yet shows that acting on a flagged answer improves accuracy. On the full test split (9,517 questions) head-selected best-of-five is +0.4 points over greedy (95% interval -0.04 to +0.84, not significant), prompted retry loses accuracy unless gated (gated gain at most +0.2 points), and the deeper pass loses 12.8 points. The head ranks candidates well, but the ranking is not yet converted into accuracy.</li>
<li><b>Grounding is a proxy.</b> "The score falls when the image is corrupted" shows the head reacts to the image, not that it tracks the specific evidence for the answer. Noise is an easy corruption.</li>
<li><b>One model size.</b> Everything is on Qwen3-VL-2B. The 4B check was not run. The exported sequential heads are single seeds.</li>
<li><b>Calibration across datasets.</b> One temperature is fitted on the pooled validation split. Per-dataset calibration was not studied.</li>
</ul>
<h3>14.3 Next steps</h3>
<ol>
<li>Find a stronger lever than choosing among five samples or retrying: the oracles leave +4.8 points (best-of-five) and +2.9 points (retry) on the table, which a better candidate generator or a latent correction could reach.</li>
<li>Run a controlled chain-of-thought condition: ask for reasoning, allow enough tokens (the deeper pass used 64), extract the final answer, score it with the head, and compare it with the deeper pass.</li>
<li>Build the latent correction step: test-time steering of the hidden state toward a higher head score, evaluated against the prompted baselines above.</li>
<li>A harder grounding test: mask only the region the answer comes from; and a softer correctness label for span answers.</li>
<li>Check the 4B model, and add extra input features to the head (several layers, image tokens).</li>
</ol>
<h3>14.4 Slide-ready summary</h3>
<table><thead><tr><th class="l">Slide</th><th class="l">Message</th><th class="l">Number to show</th></tr></thead><tbody>
<tr><td class="l">Motivation</td><td class="l">Model confidence is saturated and cannot gate</td><td class="l">1.0 on 77% to 90% of answers; AUROC @@tok@@</td></tr>
<tr><td class="l">Method</td><td class="l">Small head on frozen hidden states, before and after the answer</td><td class="l">2,049 to 2.1M parameters</td></tr>
<tr><td class="l">Result 1</td><td class="l">Works on four different datasets</td><td class="l">+0.13 to +0.27 AUROC</td></tr>
<tr><td class="l">Result 2</td><td class="l">One unified head, one pooled training</td><td class="l">AUROC @@uni@@ pooled</td></tr>
<tr><td class="l">Result 3</td><td class="l">New loss makes the score depend on the image</td><td class="l">@@gold@@ to @@gnew@@, no AUROC cost</td></tr>
<tr><td class="l">Result 4</td><td class="l">Heads can be extended without forgetting</td><td class="l">forgetting 0.011 AUROC</td></tr>
<tr><td class="l">Correction</td><td class="l">The head ranks candidates well, but selection and retry gain little so far</td><td class="l">best-of-five +0.4 points over greedy, not significant (n=9,517); oracle +4.8</td></tr>
<tr><td class="l">Next</td><td class="l">Latent correction on the hidden state</td><td class="l">not started</td></tr></tbody></table>
"""
    return fill(text, uni=fmt(uni), tok=fmt(tok), gold=fmt(g_old, 2), gnew=fmt(g_new, 2))


def section_appendix() -> str:
    """Appendix: file index, glossary, references."""
    files = [
        ["01_chartqa_original_study_report.html", "The original ChartQA study: heads, losses, the 72-run sweep, test results"],
        ["02_multidataset_qhead_report.html", "Four datasets, the unified head, metric definitions, case studies, bootstrap intervals"],
        ["03_qhead_comparison_tables.md", "Side-by-side metric tables for all scorers and datasets"],
        ["05_mixture_new_loss_report.html", "Pooled mixture, sequential training, the new loss (swap), ablation, grounding"],
        ["06_corruption_new_loss_report.html", "Noise-corrupted version of the new loss, transfer across corruptions, sequential runs"],
        ["07_correction_experiments_report.html", "Prompted retry, reranking, best-of-five, deeper pass (correction experiments)"],
        ["04_sweep_data/", "Machine-readable sweep, sequential and analysis files behind every table"],
        ["../../trained_heads/README.md", "The 540 trained heads and how to load them"],
        ["../../docs/approach1_unified_q_head.md", "Design note: datasets, unified head, loss, protocol"],
    ]
    refs = [
        "Guo, C., Pleiss, G., Sun, Y., Weinberger, K. On Calibration of Modern Neural Networks. ICML 2017 (temperature scaling).",
        "Ho, J., Jain, A., Abbeel, P. Denoising Diffusion Probabilistic Models. NeurIPS 2020 (the noising step used for the corrupted image).",
        "Masry, A., et al. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. Findings of ACL 2022.",
        "Singh, A., et al. Towards VQA Models That Can Read (TextVQA). CVPR 2019.",
        "Mathew, M., Karatzas, D., Jawahar, C. V. DocVQA: A Dataset for VQA on Document Images. WACV 2021 (ANLS).",
        "Lu, P., et al. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (ScienceQA). NeurIPS 2022.",
        "Antol, S., et al. VQA: Visual Question Answering. ICCV 2015 (VQA soft accuracy).",
    ]
    glossary = [
        ("Answer-token probability", "The geometric mean of the probabilities the model gave its own answer tokens. The untrained baseline."),
        ("Linear probe", "A trained head with no hidden layer, \\(s = w^{\\top}h + b\\)."),
        ("Pre-generation head", "Scores a question before any answer exists, from the prompt state."),
        ("Verifier head", "Scores a specific generated answer, from the answer state."),
        ("Unified head", "One network that reads both states, with a mode that works without the answer."),
        ("AUROC", "Probability that a random correct answer outranks a random wrong one."),
        ("Error AUPRC", "How well the wrong answers are packed at the top of a queue sorted by trust."),
        ("Order accuracy (grounding)", "Share of correct answers whose score falls when the image is corrupted. 0.5 means no effect."),
        ("Forgetting", "Average drop from the best score a dataset had after its own stage to its score after the last stage."),
        ("Replay", "Mixing a fixed share of earlier datasets' training rows into each later stage."),
        ("Oracle", "An unattainable upper bound that uses the ground truth to choose."),
    ]
    return f"""<h2 id="appendix">Appendix A. Files and references</h2>
<h3>A.1 Where each result lives</h3>
{table(["File in results/approach1/", "What it contains"], [[f'<a href="{name}">{name}</a>', text] for name, text in files], left_cols=(0, 1))}
<h3>A.2 Glossary</h3>
{table(["Term", "Meaning"], [[t, d] for t, d in glossary], left_cols=(0, 1))}
<h3>A.3 References</h3><ol class="refs">{"".join(f"<li>{esc(x)}</li>" for x in refs)}</ol>"""


# ------------------------------------------------------------------ page assembly



# ------------------------------------------------------------------ extra detail


def short(text, limit: int = 140) -> str:
    """One line of text, cut at a length, escaped for HTML."""
    text = " ".join(html.unescape(str(text)).split())
    return esc(text if len(text) <= limit else text[: limit - 3] + "...")


def extras_datasets(pd: dict) -> str:
    """Sections 6.5 to 6.9: calibration, the full gate table, categories, case studies, answer length."""
    panels = []
    for key, name in DATASETS:
        series = []
        for scorer in ("token_confidence", "unified_joint"):
            bins = [b for b in pd[key]["analysis"]["reliability"][scorer] if b["count"] >= 10]
            series.append((scorer, [b["mean_probability"] for b in bins], [b["accuracy"] for b in bins]))
        panels.append(line_panel(name, None, series, 0.0, 1.0, 0.25, "Mean predicted probability", "Observed accuracy", diagonal=True))
    rows = []
    for key, name in DATASETS:
        a = pd[key]["analysis"]
        for flag in (0.05, 0.1, 0.2, 0.3, 0.5):
            rows.append([name, pct(flag, 0), pct(capture_at(a, "token_confidence", flag), 0), pct(capture_at(a, "separate_verifier", flag), 0), pct(capture_at(a, "unified_joint", flag), 0)])
    cov_rows = []
    for key, name in DATASETS:
        a = pd[key]["analysis"]
        cov_rows.append([name, pct(a["splits"]["test"]["accuracy"])] + [f"{pct(accuracy_at(a, 'token_confidence', c))} / {pct(accuracy_at(a, 'unified_joint', c))}" for c in (0.9, 0.8, 0.5, 0.2)])
    cat_rows = []
    for key, name in DATASETS:
        for c in pd[key]["analysis"]["categories"]:
            cat_rows.append([name, c["category"], f"{c['count']:,}", pct(c["accuracy"]), pct(c["mean_predicted"])])
    case_rows = []
    group_label = {"caught_errors": "Caught error", "missed_errors": "Missed error", "false_alarms": "False alarm", "confident_correct": "Confident and correct"}
    for key, name in DATASETS:
        for group in group_label:
            for case in pd[key]["analysis"]["case_studies"][group][:2]:
                case_rows.append([name, group_label[group], short(case["question"], 110), short(case["reference"], 40), short(case["prediction"], 50), f"{case['joint_score']:.2f}", f"{case['token_confidence']:.2f}"])
    length_rows = []
    for key, name in DATASETS:
        a = pd[key]["analysis"]["answer_length_baseline"]
        length_rows.append([name, fmt(a["auroc"]), fmt(a["error_auprc"]), f"{a['mean_length_correct']:.1f}", f"{a['mean_length_wrong']:.1f}"])
    text = r"""
<h3>6.5 Calibration</h3>
<p>A score is useful as a probability only if it matches how often answers are right. The charts plot, for bins of predicted probability, the observed accuracy of the answers in the bin (bins with fewer than 10 answers are dropped). The dashed diagonal is perfect calibration. The answer-token probability is over-confident: in most bins it sits well below the diagonal, and answers it scores between 0.6 and 0.9 are right only about 30% to 60% of the time. The unified head follows the diagonal closely, which is why its ECE is low.</p>
@@calib@@
<h3>6.6 The gate in more detail</h3>
<p>Share of all errors found when flagging a given fraction of the least trusted answers (a random flag finds the same share as the flag rate):</p>
@@gate@@
<p>Accuracy of the answers kept when only the most trusted fraction is kept (answer-token probability / unified head):</p>
@@cov@@
@@categories@@
<h3>6.8 Examples</h3>
<p>Two examples per group per dataset from the test split. <b>Caught error</b>: a wrong answer the unified head scored low. <b>Missed error</b>: a wrong answer it scored high. <b>False alarm</b>: a correct answer it scored low. <b>Confident and correct</b>: a correct answer it scored high. The unified score is the head's probability that the answer is right.</p>
@@cases@@
<p>The caught errors show what the head is picking up: overlong or looping generations (a repeated string on a TextVQA question, a long list of buttons), chart arithmetic written out step by step, and answers that name the wrong option. Two kinds of missed error appear. Some are confident misreads, such as a wrong value read from a chart, which a head that only sees hidden states has no way to check. Others are answers that are right in spirit and marked wrong by the strict label rule ("live" for a question whose reference is "yes", "24 parts" against "24"), so part of the missed-error count reflects label noise and not head failure. The false alarms show the reverse: correct answers in an unusual format ("1 2 3 4 5 6 7 8 9 0" for "1234567890") that the head distrusts.</p>
<h3>6.9 A length shortcut check</h3>
<p>If wrong answers were simply longer, a head could score well by counting characters. An answer-length score alone (length in characters, shorter answers scored as more likely correct) reaches only the AUROCs below, where 0.5 is chance and the heads reach 0.78 to 0.92, so the heads are not using length as a shortcut. TextVQA is the one dataset where length carries some signal (wrong answers average 13.7 characters against 7.8), and the heads still beat it by more than 0.25 AUROC.</p>
@@length@@
"""
    cat_html = ""
    if cat_rows:
        cat_html = "<h3>6.7 Question types</h3><p>Where the dataset labels question types, accuracy and the unified head's mean predicted probability by type. A close match means the head knows which types are hard. The head is well matched for most types and over-confident on handwritten pages (74% accurate against 83% predicted).</p>" + table(["Dataset", "Question type", "Questions", "Accuracy", "Mean predicted probability"], cat_rows, left_cols=(0, 1))
    return fill(text, calib=figure("fig-calib", "Reliability of the answer-token probability and the unified head", f'<div class="grid2">{"".join(panels)}</div>' + legend([(SCORER_LABEL["token_confidence"], "c-tok"), (SCORER_LABEL["unified_joint"], "c-uj")]), "Test split."),
                gate=table(["Dataset", "Flag rate", "Answer-token probability", "Separate verifier", "Unified joint"], rows), cov=table(["Dataset", "Base accuracy", "Keep 90%", "Keep 80%", "Keep 50%", "Keep 20%"], cov_rows),
                categories=cat_html, cases=table(["Dataset", "Group", "Question", "Reference", "Model answer", "Unified score", "Answer-token prob."], case_rows, left_cols=(0, 1, 2, 3, 4)),
                length=table(["Dataset", "Answer-length AUROC", "Answer-length error AUPRC", "Mean length, correct", "Mean length, wrong"], length_rows))


def marginal_rows(sweep: dict, kind: str, mode: str) -> list:
    """Mean validation AUROC and error AUPRC for each value of each swept factor."""
    factors = ["architecture", "learning_rate"] + {"pre": ["rank_weight", "ground_weight"], "verifier": ["pair_weight", "ground_weight"], "unified": ["pre_weight", "pair_weight", "ground_weight"]}[kind]
    rows = []
    for factor in factors:
        values = sorted({item["config"][factor] for item in sweep["configs"]}, key=str)
        for value in values:
            items = [i for i in sweep["configs"] if i["config"][factor] == value]
            aurocs = [i["val"][mode]["correctness_auroc"] for i in items]
            auprcs = [i["val"][mode]["error_auprc"] for i in items]
            rows.append([factor.replace("_", " "), str(value), str(len(items)), f"{sum(aurocs) / len(aurocs):.4f}", f"{max(aurocs):.4f}", f"{sum(auprcs) / len(auprcs):.4f}"])
    return rows


def extras_mixture(r: Results) -> str:
    """Sections 8.4 to 8.6: the gate on the pool, hyperparameter effects, and seed spread."""
    m = r.analysis
    grid = m["grid"]
    flag, cover = grid.index(0.2), grid.index(0.5)
    rows = []
    tok = m["token_curves"]["pooled"]
    rows.append(["Answer-token probability", pct(tok["error_capture"][flag], 0), pct(tok["accuracy_at_coverage"][cover])])
    for kind, group, mode, label in (("unified", "old_linear", "joint", "Linear probe (unified joint)"), ("unified", "old", "joint", "Unified joint, old loss"), ("unified", "new", "joint", "Unified joint, new loss"), ("verifier", "old", "main", "Separate verifier, old loss"), ("pre", "old", "main", "Separate pre-generation, old loss")):
        curve = m["curves"][kind][group][mode]["pooled"]
        rows.append([label, pct(curve["error_capture"][flag], 0), pct(curve["accuracy_at_coverage"][cover])])
    hyper = ""
    for kind in KINDS:
        sweep = corr.load(f"mixture_{kind}")
        if sweep:
            mode = "main" if kind != "unified" else "joint"
            hyper += f"<h4>{KIND_NAME[kind]}</h4>" + table(["Factor", "Value", "Configurations", "Mean validation AUROC", "Best validation AUROC", "Mean validation error AUPRC"], marginal_rows(sweep, kind, mode), left_cols=(0, 1))
    seed_rows = []
    for kind, mode in (("pre", "main"), ("verifier", "main"), ("unified", "pre_generation"), ("unified", "joint")):
        for group, label in (("old", "old loss"), ("new", "new loss")):
            node = r.metric(kind, group, mode, "correctness_auroc")
            if node:
                seed_rows.append([KIND_NAME[kind] + ("" if kind != "unified" else f", {mode.replace('_', '-')}"), label, f"{node[0]:.4f}", f"{node[1]:.4f}"])
    text = r"""
<h3>8.4 The gate on the pooled mixture</h3>
<p>Flagging the 20% of pooled answers with the lowest score, and keeping the most trusted 50%:</p>
@@gate@@
<h3>8.5 What the sweep says about the hyperparameters</h3>
<p>Every sweep fits many configurations, so it is also a controlled experiment on the design choices. Each row averages the validation metric over all configurations that share one value of one factor (with the other factors varying), so it shows the marginal effect of that choice. These are validation numbers from the swap sweep; the noise sweep is in the Excel workbook.</p>
@@hyper@@
<p>Three things stand out. First, the learning rate barely matters (under 0.003 AUROC). Second, the architecture matters a little: the linear probe averages 0.007 to 0.013 AUROC below the MLPs, and MLP-256 and MLP-512 are indistinguishable, which is why the linear probe is so close to the larger heads in the winners. Third, the grounding weight \(\lambda\) and the pair weight are the factors that change the picture: a pair weight of 0.25 lowers mean validation AUROC by 0.013 (unified) to 0.027 (verifier), while \(\lambda\) costs 0.003 or less at 0.1 and up to 0.007 at 0.5. This is why the winning new-loss configurations use \(\lambda = 0.1\) and no pair term.</p>
<h3>8.6 How much does a seed matter?</h3>
<p>Standard deviation of the pooled test AUROC across the three seeds of each winner. Differences smaller than about twice these values should not be read as real.</p>
@@seeds@@
"""
    return fill(text, gate=table(["Scorer", "Errors found in the lowest 20%", "Accuracy of the most trusted 50%"], rows), hyper=hyper, seeds=table(["Head", "Loss", "Pooled AUROC", "Std over seeds"], seed_rows, left_cols=(0, 1))) + heads_vs_probes_section()



def heads_vs_probes_section() -> str:
    """Section 8.7: trained heads against linear probes, per dataset, in both training settings."""
    from experiments.approach1_latent_self_correction import heads_vs_probes as hp
    rows = hp.comparison_rows()
    parts = []
    summary = []
    for setting in ("Trained on the pooled mixture", "Trained on each dataset alone"):
        subset = [r for r in rows if r["setting"] == setting]
        table_rows = []
        for r in subset:
            table_rows.append([r["dataset"], r["head_label"], fmt(r["token_auroc"]), f"{r['linear_correctness_auroc']:.3f} ± {r['linear_correctness_auroc_std']:.3f}", f"{r['head_correctness_auroc']:.3f} ± {r['head_correctness_auroc_std']:.3f}",
                               f"{r['diff_correctness_auroc']:+.3f}", f"{r['head_arch']} ({r['head_params']:,})", f"{r['linear_error_auprc']:.3f}", f"{r['head_error_auprc']:.3f}", hp.verdict(r)])
        better = sum(1 for r in subset if hp.verdict(r) == "head better")
        same = sum(1 for r in subset if hp.verdict(r) in ("within seed spread", "same model (the best head is linear)"))
        worse = sum(1 for r in subset if hp.verdict(r) == "linear probe better")
        gains = [r["diff_correctness_auroc"] for r in subset]
        share = sorted((r["linear_correctness_auroc"] - r["token_auroc"]) / (r["head_correctness_auroc"] - r["token_auroc"]) for r in subset if r["head_correctness_auroc"] > r["token_auroc"])
        summary.append(f"<li><b>{setting}.</b> The trained head is better than the linear probe by more than the seed spread in {better} of {len(subset)} comparisons, within the spread (or the same model) in {same}, and worse in {worse}. "
                       f"The mean AUROC difference is {sum(gains) / len(gains):+.4f} and the largest is {max(gains):+.3f}. The linear probe delivers {100 * share[0]:.0f}% to {100 * share[-1]:.0f}% (median {100 * share[len(share) // 2]:.0f}%) of the head's gain over the answer-token probability.</li>")
        parts.append(f"<h4>{setting}</h4>" + table(["Dataset", "Head", "Answer-token prob. AUROC", "Linear probe AUROC", "Trained head AUROC", "Difference", "Head architecture (parameters)", "Linear probe error AUPRC", "Trained head error AUPRC", "Reading"], table_rows, left_cols=(0, 1, 6, 9)))
    chart_rows = []
    for r in rows:
        if r["setting"] == "Trained on the pooled mixture" and r["kind"] == "unified" and r["mode"] == "joint":
            chart_rows.append((r["dataset"], {"token": r["token_auroc"], "linear": r["linear_correctness_auroc"], "head": r["head_correctness_auroc"]}))
    series = [("token", SCORER_LABEL["token_confidence"], "c-tok"), ("linear", "Linear probe", "c-sp"), ("head", "Trained head", "c-uj")]
    chart = dot_plot(chart_rows, series, 0.6, 1.0, "Correctness AUROC on the test split (axis starts at 0.6)", caption_id="fig-hvl") + legend([(label, css) for _, label, css in series])
    text = r"""
<h3>8.7 Trained heads against linear probes, per dataset</h3>
<p>The question here is how much the extra capacity buys. A <b>linear probe</b> is the best linear head of a sweep: no hidden layer, 2,049 parameters for a separate head and 12,290 for the unified head. The <b>trained head</b> is the best head of the same sweep over all three architectures (linear, MLP-256, MLP-512), and it is a linear probe whenever the validation score prefers one. Both are chosen on validation with the old loss and scored on the test split, mean and standard deviation over three seeds. The comparison is made in the two training settings used in this report, for every head kind and mode. A difference is called real only if it is larger than the combined seed spread of the two. The answer-token probability is shown as a reference.</p>
@@chart@@
<ul>@@summary@@</ul>
@@tables@@
<div class="callout"><b class="t">Reading the comparison</b>The trained heads are better than linear probes, but only a little. Trained on the pool, the head is ahead in all 16 comparisons by 0.001 to 0.013 AUROC (mean 0.006), and the lead is larger than the seed spread in 12 of them; the largest leads are the question-only head on DocVQA (+0.013), the verifier on ChartQA (+0.010) and the question-only head on TextVQA (+0.010). Trained on each dataset alone, with 6,000 to 28,000 training rows, the mean lead falls to 0.003: the head wins by more than the seed spread in 7 comparisons, ties in 7 (in two of them the best head is a linear probe, the unified head on ChartQA), and loses in 2, both for the unified head on TextVQA (-0.002 and -0.006). A larger head therefore needs the pooled data to pay off consistently. Either way the linear probe delivers about 90% to 99% of the head's gain over the answer-token probability (above 100% where the probe is better) with well under 1% of the parameters of the larger head, which makes it the sensible default and the larger heads a small refinement.</div>
"""
    return fill(text, chart=figure("fig-hvl", "Unified head, joint mode: answer-token probability, linear probe and trained head (trained on the pool)", chart, "Each row is one dataset's test rows."), summary="".join(summary), tables="".join(parts))

def image_tag(data: str) -> str:
    """An inline thumbnail from base64 JPEG data."""
    return f'<img class="thumb" alt="" src="data:image/jpeg;base64,{data}">' if data else ""


def extras_losses(swap: dict, noise: dict) -> str:
    """Sections 9.9 to 9.11: the corrupted images, the effect of the grounding weight, all head kinds per dataset."""
    gallery_swap = json.loads((SWEEP / "mixture_analysis.json").read_text(encoding="utf-8"))["gallery"]
    gallery_noise = {g["dataset"]: g for g in json.loads((SWEEP / "corruption_mixture_noise_analysis.json").read_text(encoding="utf-8"))["gallery"]}
    cards = ""
    for item in gallery_swap:
        noised = gallery_noise.get(item["dataset"])
        cards += (f'<div class="card" style="margin:0"><h4 style="margin-top:0">{NAMES.get(item["dataset"], item["dataset"])}</h4><div class="grid2" style="grid-template-columns:1fr 1fr 1fr">'
                  f'<figure style="margin:0"><figcaption class="small">Clean</figcaption>{image_tag(item["clean"])}</figure>'
                  f'<figure style="margin:0"><figcaption class="small">Swapped</figcaption>{image_tag(item["corrupt"])}</figure>'
                  f'<figure style="margin:0"><figcaption class="small">Noised (alpha 0.15)</figcaption>{image_tag(noised["corrupt"]) if noised else ""}</figure></div></div>')
    lam_rows = []
    for label, tag in (("swap", "mixture"), ("noise", "mixture_noise")):
        for kind in ("verifier", "unified"):
            sweep = corr.load(f"{tag}_{kind}")
            if not sweep:
                continue
            mode = "main" if kind == "verifier" else "joint"
            for value in (0.0, 0.1, 0.5):
                items = [i for i in sweep["configs"] if i["config"]["ground_weight"] == value]
                aurocs = [i["val"][mode]["correctness_auroc"] for i in items]
                lam_rows.append([label, KIND_NAME[kind], f"{value:g}", str(len(items)), f"{sum(aurocs) / len(aurocs):.4f}", f"{max(aurocs):.4f}"])
    all_rows = []
    for key, name in (("chartqa", "ChartQA"), ("textvqa", "TextVQA"), ("docvqa", "DocVQA"), ("scienceqa", "ScienceQA")):
        for kind in KINDS:
            sw, nz = corr.load(f"{key}_swap_{kind}"), corr.load(f"{key}_noise_{kind}")
            if sw and nz:
                all_rows.append([name, KIND_NAME[kind], fmt(corr.auroc(sw["winners"]["old"], kind)), fmt(corr.auroc(sw["winners"]["new"], kind)), fmt(corr.auroc(nz["winners"]["new"], kind)),
                                 fmt(corr.grounding_against(nz, "old", kind, "noise")), fmt(corr.own_grounding(sw["winners"]["new"], kind)), fmt(corr.own_grounding(nz["winners"]["new"], kind))])
    text = r"""
<h3>9.9 What the corrupted images look like</h3>
<p>The same clean test image, the swapped partner (another example's image from the same split), and the noised version used by the noise loss. The noised image keeps the layout but loses most of the detail, which is why a head cannot satisfy the noise term by detecting a question and image mismatch. Note that for ScienceQA the swapped partner is another map of a similar kind, so the swap is a milder corruption there than on the other datasets, which may be part of why grounding is lowest on ScienceQA with swap.</p>
<div class="grid2">@@cards@@</div>
<h3>9.10 The grounding weight</h3>
<p>The grounding weight \(\lambda\) trades ranking quality for grounding. Mean validation AUROC over all configurations that use each value of \(\lambda\) (swap and noise sweeps):</p>
@@lam@@
<p>Going from \(\lambda = 0\) to 0.1 changes mean validation AUROC by 0.003 or less (up for the swap unified head, down for the others); going to 0.5 costs 0.001 to 0.007. The sweep therefore picks 0.1, and the grounding gains reported in this document are at that setting.</p>
<h3>9.11 All head kinds on each dataset alone</h3>
<p>The same comparison as in Section 9.8 for the pre-generation head, the verifier and the unified head.</p>
@@all@@
"""
    return fill(text, cards=cards, lam=table(["Corruption", "Head", r"\(\lambda\)", "Configurations", "Mean validation AUROC", "Best validation AUROC"], lam_rows, left_cols=(0, 1)),
                all=table(["Dataset", "Head", "AUROC, old", "AUROC, new (swap)", "AUROC, new (noise)", "Falls under noise, old", "Falls under swap, new (swap)", "Falls under noise, new (noise)"], all_rows, left_cols=(0, 1)))


def extras_sequential(r: Results) -> str:
    """Sections 10.5 and 10.6: the other dataset orders and the per-dataset forgetting."""
    maps = ""
    drop_rows = []
    if "unified" in r.continual:
        variants = r.continual["unified"]["variants"]
        for order in ("reverse", "shuffled"):
            for key, label in (("old_replay0", "old loss"), ("new_replay0", "new loss")):
                entry = variants[key]["orders"][order]
                names = [NAMES.get(n, n) for n in entry["order"]]
                maps += figure(f"fig-seq-{order}-{key}", f"Unified head, {label}, {order} order", heatmap(entry["matrix"]["joint"], [f"After {n}" for n in names], [NAMES.get(n, n) for n in r.datasets], f"fig-seq-{order}-{key}"), "")
        for key, label in (("old_replay0", "Old loss"), ("new_replay0", "New loss"), ("old_replay0.1", "Old loss, 10% replay"), ("new_replay0.1", "New loss, 10% replay")):
            entry = variants[key]["orders"]["forward"]
            matrix = entry["matrix"]["joint"]
            for position, dataset in enumerate(entry["order"][:-1]):
                column = r.datasets.index(dataset)
                after_own, final, best = matrix[position][column], matrix[-1][column], max(row[column] for row in matrix[position:])
                drop_rows.append([label, NAMES.get(dataset, dataset), fmt(after_own), fmt(final), f"{best - final:+.3f}"])
    text = r"""
<h3>10.5 The other dataset orders</h3>
<p>The forward order (ChartQA, TextVQA, DocVQA, ScienceQA) puts the largest dataset first. Two further orders let you see whether the pattern depends on it: reverse (ScienceQA first) and a shuffled order. The tables above average the three orders; here are the stage matrices for the unified head in the other two.</p>
<div class="grid2">@@maps@@</div>
<h3>10.6 What each dataset loses, one by one</h3>
<p>For the unified head in the forward order: the AUROC on each earlier dataset right after its own stage, at the end, and the drop from the best score it had after its own stage. The last dataset is not listed because nothing came after it.</p>
@@drops@@
<p>Without replay the drops are 0.008 to 0.021 and are largest for the dataset trained first (ChartQA), which has been through the most later stages. With a replay buffer the unified head's drops in this order are larger (0.022 to 0.038), so replay does not help the unified head here, which matches the sequential tables in Section 10.2 where replay leaves its forgetting unchanged or slightly higher. Replay helps the separate verifier head, not the unified head.</p>
"""
    return fill(text, maps=maps, drops=table(["Variant", "Dataset", "AUROC after its own stage", "AUROC at the end", "Drop from its best"], drop_rows, left_cols=(0, 1)))


def extras_engineering() -> str:
    """Section 13.1: measured run times."""
    rows = [
        ["Corrupted-image features, ChartQA train (28,299 questions)", "82.5 min"],
        ["Pooled sweep, pre-generation head (36 configurations x 3 seeds)", "3.4 min"],
        ["Pooled sweep, verifier head (36 x 3)", "8.0 min"],
        ["Pooled sweep, unified head (72 x 3)", "11.6 min"],
        ["Sequential training, pre-generation head (4 variants x 3 orders x 3 seeds)", "1.5 min"],
        ["Sequential training, verifier and unified heads", "3.4 min each"],
        ["Export of the 48 pooled-mixture winner heads (retraining)", "4.3 min"],
        ["Original per-dataset pipeline, TextVQA: build / baseline / features", "49.7 / 24.6 / 28.4 min"],
        ["Original per-dataset pipeline, DocVQA: build / baseline / features", "67.9 / 39.7 / 62.8 min"],
        ["Original per-dataset pipeline, ScienceQA: build / baseline / features", "2.7 / 8.0 / 11.8 min"],
    ]
    return "<h3>13.1 Measured run times</h3><p>Wall-clock times from Slurm accounting, one GPU each. The model passes (features, baselines) dominate the cost; every head experiment on the cached features is short.</p>" + table(["Job", "Time"], rows, left_cols=(0,))


def workbook_section(workbook_path: Path) -> str:
    """Appendix A.4: the Excel workbook and its sheets."""
    from openpyxl import load_workbook
    sheet = load_workbook(workbook_path, read_only=True)["Index"]
    rows = [[row[0], row[1], row[2]] for row in sheet.iter_rows(min_row=2, values_only=True)]
    return ('<h3>A.4 The Excel workbook</h3><p>Every table in this report and the data behind it is in <code>approach1_results.xlsx</code>, one sheet per table. Download it with the button at the top of the report or from <code>results/approach1/08_approach1_results.xlsx</code>. The sheets are:</p>'
            + table(["Sheet", "What it holds", "Rows"], rows, left_cols=(0, 1)))


def katex_inline() -> str:
    """KaTeX CSS with fonts embedded as data URIs, and the scripts, so the page needs no network."""
    css = (ASSETS / "katex.min.css").read_text(encoding="utf-8")
    pattern = re.compile(r'url\((fonts/[^)]+?\.woff2)\) format\("woff2"\),url\(fonts/[^)]+?\.woff\) format\("woff"\),url\(fonts/[^)]+?\.ttf\) format\("truetype"\)')

    def embed(match):
        data = base64.b64encode((ASSETS / match.group(1)).read_bytes()).decode("ascii")
        return f'url(data:font/woff2;base64,{data}) format("woff2")'

    css, count = pattern.subn(embed, css)
    if count == 0:
        raise ValueError("no KaTeX font rules were rewritten")
    scripts = (ASSETS / "katex.min.js").read_text(encoding="utf-8") + "\n" + (ASSETS / "auto-render.min.js").read_text(encoding="utf-8")
    return f"<style>{css}</style><script>{scripts}</script>"


def build() -> str:
    """Assemble the whole report."""
    r = Results()
    pd = per_dataset()
    swap = {k: corr.load(f"mixture_xeval_{k}") for k in KINDS}
    base_swap = {k: corr.load(f"mixture_{k}") for k in KINDS}
    noise = {k: corr.load(f"mixture_noise_{k}") for k in KINDS}
    xeval = swap
    swap_main = {k: base_swap[k] if base_swap[k] else swap[k] for k in KINDS}
    data = correction_tables()
    uni = r.metric("unified", "old", "joint", "correctness_auroc")[0]
    tok = r.baseline("pooled")["token_confidence"]["correctness_auroc"]
    kpis = [
        (f"{tok:.3f} to {uni:.3f}", "Pooled AUROC, answer-token probability to unified head"),
        ("0.750 to 0.895", "ChartQA AUROC, original study, probability to linear probe"),
        ("+0.13 to +0.27", "AUROC gain of the learned heads on every dataset"),
        ("0.011", "AUROC forgotten by the unified head over four datasets"),
        (f"{corr.grounding_against(noise['unified'], 'old', 'unified', 'noise'):.2f} to {corr.grounding_against(noise['unified'], 'new', 'unified', 'noise'):.2f}", "Share of correct answers whose score falls under noise, old to new loss"),
        ("540", "Trained heads in the repository, 2,049 to 2.1 million parameters"),
    ]
    items = [("overview", "1 Road map"), ("foundations", "2 Foundations"), ("baseline", "3 Baseline"), ("original", "4 ChartQA study"), ("sizes", "5 Head sizes"), ("datasets", "6 Datasets"), ("unified", "7 Unified head"),
             ("mixture", "8 Mixture"), ("losses", "9 Losses"), ("seq", "10 Forgetting"), ("heads", "11 Trained heads"), ("correction", "12 Correction"), ("engineering", "13 Engineering"), ("meaning", "14 Meaning"), ("appendix", "Appendix")]
    from experiments.approach1_latent_self_correction import export_results_workbook as workbook
    workbook_path = workbook.build_workbook(RESULTS / "approach1_results.xlsx")
    encoded = base64.b64encode(workbook_path.read_bytes()).decode("ascii")
    download = (f'<a class="btn" download="approach1_results.xlsx" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded}">'
                f'Download all results tables (Excel, {len(workbook_path.read_bytes()) // 1000} KB)</a>')
    body = "".join([
        cover(r, pd, kpis).replace("<div class=\"kpi\">", download + "<div class=\"kpi\">", 1), toc(items), section_overview(r), section_foundations(r, pd), section_baseline(r, pd), section_original(r), section_sizes(r, pd),
        section_datasets(r, pd), extras_datasets(pd), section_unified(r, pd), section_mixture(r), extras_mixture(r), section_losses(r, swap_main, noise, xeval), extras_losses(swap_main, noise), section_sequential(r), extras_sequential(r),
        section_heads(), section_correction(data), section_correction_full(full_correction_data()), section_engineering(), extras_engineering(), section_meaning(r, swap_main, noise), section_appendix(), workbook_section(workbook_path),
    ])
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Approach 1 Complete Report</title>'
            f"<style>{CSS}{EXTRA_CSS}{BAR_CSS}</style>{katex_inline()}</head><body><main>{body}</main>{KATEX_START}</body></html>")


def export_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Print the page to PDF with a headless browser, waiting for the math to render."""
    for candidate in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(candidate).exists():
            command = [candidate, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
            subprocess.run(command, check=False, capture_output=True, timeout=240)
            return pdf_path.exists()
    return False


def main() -> None:
    """Build the report and optionally export a PDF."""
    parser = argparse.ArgumentParser(description="Build the complete Approach 1 report.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pdf", action="store_true", help="Also print a PDF with a headless browser.")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(), encoding="utf-8")
    print(f"[report] wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    if args.pdf:
        pdf = args.output.with_suffix(".pdf")
        print(f"[pdf] {'wrote ' + str(pdf) if export_pdf(args.output, pdf) else 'no headless browser found'}")


if __name__ == "__main__":
    main()
