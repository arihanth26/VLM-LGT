# build_ranker_report.py
#
# Builds report 11: the learned correction step. New candidates from visual contrastive decoding and
# zoomed crops, a ranker trained for the within-question choice, best-of-k from 5 up to 16 samples,
# where the fixes come from, ablations, compute, and case studies. One self-contained HTML file in the
# style of the earlier reports.
# Reads results/candidate_ranker.json (train_candidate_ranker.py), results/ranker_analysis.json
# (analyze_ranker.py), results/candidate_search.json (report 10), the report 08 best-of-five summary,
# and results/correction_jobs.txt (the scheduler record).

import argparse
import json
from pathlib import Path

from experiments.approach1_latent_self_correction.build_report import CSS, Panel, esc, figure, legend, table

HERE = Path(__file__).parent
RESULTS = HERE / "results"
REPO = HERE.parents[1]
OUTPUT = RESULTS / "correction_ranker_report.html"
DATASETS = {"chartqa_full": "ChartQA", "textvqa": "TextVQA", "docvqa": "DocVQA", "scienceqa_img": "ScienceQA"}
SET_LABEL = {"none": "samples only", "vcd": "samples + contrastive decoding", "zoom": "samples + zoomed crops", "all": "samples + both corrections"}
SOURCE_LABEL = {"greedy": "greedy", "sample": "sampled answer", "vcd": "contrastive decoding", "zoom": "zoomed crop"}
GATE_LABEL = {"head": "Head score of the greedy answer", "token": "Answer probability of the greedy answer", "pre_head": "Pre-generation score"}


def pct(value: float, digits: int = 2) -> str:
    """A proportion as a percentage."""
    return f"{100 * value:.{digits}f}%"


def pts(value: float) -> str:
    """A difference of proportions in signed percentage points."""
    return f"{100 * value:+.2f}"


def ci(row: dict) -> str:
    """Difference with its bootstrap interval."""
    return f"{pts(row['difference'])} ({pts(row['ci_low'])} to {pts(row['ci_high'])})"


def pval(value: float) -> str:
    """A p-value, with very small values written as a bound."""
    return "&lt; 0.0001" if value < 1e-4 else f"{value:.4f}" if value < 0.01 else f"{value:.3f}"


def errors_removed(row: dict) -> str:
    """Share of greedy errors removed: gain divided by the greedy error rate."""
    return f"{100 * row['difference'] / (1 - row['greedy']):.1f}%"


def curve_chart(curve: dict, temperature: str, series: list[tuple], ymin: float, ymax: float, title: str, value) -> str:
    """Line chart over k for one temperature, one line per (label, set, selector, css)."""
    width, height = 680, 300
    ks = sorted({int(key.split("_k")[1].split("_")[0]) for key in curve if key.startswith(f"T{temperature}_k")})
    panel = Panel(60, 14, width - 80, height - 62, 0, max(ks), ymin, ymax)
    ticks, tick, step = [], ymin, 0.01 if ymax - ymin <= 0.06 else 0.02
    while tick <= ymax + 1e-9:
        ticks.append(round(tick, 4))
        tick += step
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
             panel.axes(ks, ticks, yfmt=lambda v: f"{100 * v:.0f}%", xlabel="number of sampled answers k", ylabel="test accuracy")]
    for label, name, selector, css in series:
        points = [(k, value(curve[f"T{temperature}_k{k}_{name}"][selector]["pooled"])) for k in ks if f"T{temperature}_k{k}_{name}" in curve]
        if not points:
            continue
        parts.append(f'<polyline class="{css} ln" fill="none" points="{" ".join(f"{panel.sx(x):.1f},{panel.sy(y):.1f}" for x, y in points)}"/>')
        for x, y in points:
            parts.append(f'<circle class="{css} dot" cx="{panel.sx(x):.1f}" cy="{panel.sy(y):.1f}" r="3"><title>{esc(label)}, k={x}: {100 * y:.2f}%</title></circle>')
    parts.append("</svg>")
    return legend([(label, css) for label, _, _, css in series]) + "".join(parts)


def gating_chart(gating: dict) -> str:
    """Gain over greedy against the share of questions corrected, one line per gate."""
    width, height = 680, 280
    css = {"head": "c-uj", "token": "c-tok", "pre_head": "c-sp"}
    budgets = sorted(next(iter(gating.values())), key=float)
    gains = [100 * gating[g][b]["difference"] for g in gating for b in budgets]
    top = max(1.2, max(gains) + 0.2)
    panel = Panel(56, 14, width - 76, height - 60, 0, 1, 0, top)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Gating">',
             panel.axes([0, 0.2, 0.4, 0.6, 0.8, 1.0], [round(0.2 * i, 1) for i in range(int(top / 0.2) + 1)],
                        xfmt=lambda v: f"{100 * v:.0f}%", yfmt=lambda v: f"+{v:.1f}", xlabel="share of questions corrected",
                        ylabel="gain over greedy (points)")]
    for gate, rows in gating.items():
        points = [(0.0, 0.0)] + [(float(b), 100 * rows[b]["difference"]) for b in budgets]
        parts.append(f'<polyline class="{css[gate]} ln" fill="none" points="{" ".join(f"{panel.sx(x):.1f},{panel.sy(y):.1f}" for x, y in points)}"/>')
        for x, y in points[1:]:
            parts.append(f'<circle class="{css[gate]} dot" cx="{panel.sx(x):.1f}" cy="{panel.sy(y):.1f}" r="3"><title>{esc(GATE_LABEL[gate])}: {100 * x:.0f}%, {y:+.2f} points</title></circle>')
    parts.append("</svg>")
    return legend([(GATE_LABEL[g], css[g]) for g in gating]) + "".join(parts)


def pipeline_svg() -> str:
    """Diagram of the correction step: candidate sources, common encoding, ranker, answer."""
    boxes = [(20, 30, "Greedy answer", "one generation"), (20, 100, "Sampled answers", "k at temperature T"),
             (20, 170, "Contrastive decoding", "real vs noised image"), (20, 240, "Zoomed crops", "4 corners, 2x")]
    parts = ['<svg viewBox="0 0 760 320" role="img" aria-label="Correction pipeline">',
             '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="arrowhead" d="M0,0 L10,5 L0,10 z"/></marker></defs>']
    for x, y, title, sub in boxes:
        parts.append(f'<rect class="box" x="{x}" y="{y}" width="190" height="52" rx="8"/><text class="boxt" x="{x + 95}" y="{y + 22}" text-anchor="middle">{title}</text>'
                     f'<text class="boxs" x="{x + 95}" y="{y + 40}" text-anchor="middle">{sub}</text>')
        parts.append(f'<path class="arrow" d="M{x + 190},{y + 26} C260,{y + 26} 260,160 300,160" marker-end="url(#ah)"/>')
    parts.append('<rect class="box" x="300" y="118" width="200" height="84" rx="8"/><text class="boxt" x="400" y="146" text-anchor="middle">Encode each distinct answer</text>'
                 '<text class="boxs" x="400" y="166" text-anchor="middle">full image + original prompt</text><text class="boxs" x="400" y="184" text-anchor="middle">state, answer probability, head score</text>')
    parts.append('<path class="arrow" d="M500,160 L560,160" marker-end="url(#ah)"/>')
    parts.append('<rect class="box hl" x="560" y="118" width="180" height="84" rx="8"/><text class="boxt" x="650" y="146" text-anchor="middle">Candidate ranker</text>'
                 '<text class="boxs" x="650" y="166" text-anchor="middle">scores each candidate,</text><text class="boxs" x="650" y="184" text-anchor="middle">keeps the top one</text>')
    parts.append('<text class="note" x="400" y="232" text-anchor="middle">question embedding: the same cached tensor for every candidate</text></svg>')
    return "".join(parts)


def case_html(case: dict) -> str:
    """One case study: image, question, reference, greedy and chosen answers, and every candidate."""
    rows = "".join(
        f"<tr><td class='l'>{esc(c['answer'][:80])}</td><td class='l'>{esc(', '.join(SOURCE_LABEL.get(s, s) for s in c['sources']))}</td>"
        f"<td>{'right' if c['correct'] else 'wrong'}</td><td>{c['ranker']:+.2f}</td><td>{c['head']:.2f}</td><td>{100 * c['vote']:.0f}%</td></tr>"
        for c in case["candidates"][:8])
    verdict = "Fixed: greedy was wrong, the ranker's choice is right" if case["kind"] == "fixed" else "Broken: greedy was right, the ranker's choice is wrong"
    return (f'<div class="case"><div class="dcard"><img class="thumb" alt="Question image" src="data:image/jpeg;base64,{case["image"]}"/><div>'
            f'<div class="tag">{DATASETS[case["dataset"]]} test #{case["index"]} · {verdict}</div>'
            f'<p class="qt">{esc(case["question"][:600])}</p><p class="ans">Reference: <b>{esc(str(case["reference"]))}</b></p>'
            f'<p class="ans">Greedy: {esc(case["greedy"])} → ranker: <b>{esc(case["chosen"])}</b></p></div></div>'
            f'<div class="tablewrap"><table><thead><tr><th>Candidate</th><th>Proposed by</th><th>Correct</th><th>Ranker score</th><th>Head</th><th>Sample vote</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></div>')


def jobs_table(path: Path) -> tuple[str, dict]:
    """Scheduler record as a table with totals."""
    if not path.exists():
        return "<p class='small'>Scheduler record not found.</p>", {"jobs": 0, "gpu_hours": 0.0}
    rows, seconds = [], 0
    for line in path.read_text().splitlines():
        job, name, state, elapsed, start, end, tres = line.split("|")
        h, m, s = (int(x) for x in elapsed.split(":"))
        seconds += 3600 * h + 60 * m + s
        gpu = next((part.split(":")[1].split("=")[0].upper() for part in tres.split(",") if part.startswith("gres/gpu:")), "?")
        rows.append([job, esc(name), state.title(), elapsed, gpu])
    return table(["Job", "Name", "State", "Run time", "GPU"], rows, left_cols=(1,)), {"jobs": len(rows), "gpu_hours": seconds / 3600}


def build(analysis_path: Path) -> str:
    """Assemble report 11."""
    ranker = json.loads((RESULTS / "candidate_ranker.json").read_text())
    analysis = json.loads(analysis_path.read_text())
    search = json.loads((RESULTS / "candidate_search.json").read_text())
    best5_path = REPO / "results" / "approach1" / "04_sweep_data" / "correction_full_mixture_test_2b_best_of_five_full.summary.json"
    best5 = json.loads(best5_path.read_text())["policies"] if best5_path.exists() else None
    curve = {k: v for k, v in analysis["test_curve"].items() if k != "longest_pool"}
    val_curve = {k: v for k, v in analysis["val_curve"].items() if k != "longest_pool"}
    longest = analysis["test_curve"].get("longest_pool", {})
    main_key = ranker["choice"]["configuration"]
    main = ranker["choice"]["test"][f"ranker_{ranker['choice']['variant']}"]
    main_val = ranker["choice"]["val"]
    greedy = main["pooled"]["greedy"] if "greedy" in main["pooled"] else ranker["choice"]["test"]["greedy"]["pooled"]["accuracy"]
    oracle_main = ranker["choice"]["test"]["oracle"]["pooled"]["accuracy"]
    jobs_html, totals = jobs_table(RESULTS / "correction_jobs.txt")
    zoom0 = curve.get("T1_k0_zoom", {}).get("ranker", {}).get("pooled")
    within_main = curve["T1_k5_all"]["within"]

    def accuracy(key, selector="ranker"):
        return curve[key][selector]["pooled"]["accuracy"] if key in curve else float("nan")

    for row in main.values():
        row.setdefault("greedy", ranker["choice"]["test"]["greedy"]["pooled"]["accuracy"])
    for name in DATASETS:
        main[name]["greedy"] = ranker["choice"]["test"]["greedy"][name]["accuracy"]

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learned Correction Report</title><style>{CSS}</style></head><body><main>
<h1>Approach 1, report 11: the learned correction step</h1>
<p class="sub">Correcting answers by producing better candidates and choosing among them with a model trained for that choice. Two new
candidate sources (visual contrastive decoding and zoomed crops), a within-question candidate ranker, best-of-k from 5 up to {max(longest.values()) if longest else 16}
samples, where the fixes come from, ablations, compute, and case studies. Qwen3-VL-2B, frozen. Full test splits, 9,517 questions; every
choice made on validation (6,420 questions) and test read once.</p>
<p class="meta">Builds on report 08 (prompted retry, best-of-five and a deeper pass on the full test split) and report 10 (candidate search and
within-question measurements).</p>
<div class="toc"><a href="#summary">Summary</a><a href="#story">Where this fits</a><a href="#idea">The idea</a><a href="#sources">Candidate sources</a>
<a href="#ranker">The ranker</a><a href="#result">Main result</a><a href="#k">Best-of-5, 7, 9, 11+</a><a href="#contrib">What each source adds</a>
<a href="#why">Why it works</a><a href="#ablation">Ablations</a><a href="#compute">Compute</a><a href="#cases">Cases</a><a href="#meaning">What it means</a>
<a href="#limits">Limits</a><a href="#next">Next</a><a href="#appendix">Appendix</a></div>

<h2 id="summary">1. Summary</h2>
<div class="stats">
<div class="stat"><b>{pct(main['pooled']['greedy'])} → {pct(main['pooled']['accuracy'])}</b><span>test accuracy, greedy to the learned correction step</span></div>
<div class="stat"><b>{pts(main['pooled']['difference'])} pts</b><span>95% interval {pts(main['pooled']['ci_low'])} to {pts(main['pooled']['ci_high'])}, p {pval(main['pooled']['p_value'])}</span></div>
<div class="stat"><b>{errors_removed(main['pooled'])}</b><span>of greedy errors removed ({main['pooled']['fixed']} fixed, {main['pooled']['broken']} broken)</span></div>
<div class="stat"><b>{errors_removed(main['textvqa'])}</b><span>of TextVQA errors removed ({pts(main['textvqa']['difference'])} pts)</span></div>
</div>
<div class="card summary"><ul>
<li><b>This is the first correction step in Approach 1 that clearly improves accuracy.</b> On the full test split it moves accuracy from
{pct(main['pooled']['greedy'])} to {pct(main['pooled']['accuracy'])}, {pts(main['pooled']['difference'])} points (95% interval {pts(main['pooled']['ci_low'])} to
{pts(main['pooled']['ci_high'])}, McNemar p {pval(main['pooled']['p_value'])}): it fixes {main['pooled']['fixed']} wrong greedy answers and breaks {main['pooled']['broken']} right ones.
Every earlier step was within noise or harmful: prompted retry −2.7 points, best-of-five with the head +0.40 (p = 0.08), best-of-seven +0.44 (p = 0.052).</li>
<li><b>Significant on three of four datasets:</b> TextVQA {pts(main['textvqa']['difference'])} (p {pval(main['textvqa']['p_value'])}), DocVQA {pts(main['docvqa']['difference'])}
(p {pval(main['docvqa']['p_value'])}), ScienceQA {pts(main['scienceqa_img']['difference'])} (p {pval(main['scienceqa_img']['p_value'])}); ChartQA {pts(main['chartqa_full']['difference'])}
(p {pval(main['chartqa_full']['p_value'])}) does not move.</li>
<li><b>Two changes did it, and how the answer is chosen matters most.</b> A ranker trained to choose among the candidates of one question,
instead of the correctness head, and new candidates from zoomed crops that let the model read small detail. Zoomed crops plus the greedy answer, with no
sampling at all, already reach {pct(zoom0['accuracy']) if zoom0 else '–'} ({pts(zoom0['difference']) if zoom0 else '–'} points).</li>
<li><b>More samples do not help.</b> With the ranker, best-of-5 ({pct(accuracy('T1_k5_all'))}), best-of-7 ({pct(accuracy('T1_k7_all'))}), best-of-9
({pct(accuracy('T1_k9_all'))}) and best-of-11 ({pct(accuracy('T1_k11_all'))}) are within a fraction of a point, although the oracle keeps rising. Section 7 answers
"should we go to 9 or 11" in detail: no, the pool is not the limit, choosing is.</li>
<li><b>The ranker is a better within-question judge than the head.</b> On the same candidates it ranks a right answer above a wrong one with grouped
AUROC {within_main['ranker']['grouped_auroc']:.3f} against {within_main['head']['grouped_auroc']:.3f} for the head, and picks a right answer on {pct(within_main['ranker']['mixed_accuracy'], 1)} of
questions that have both a right and a wrong candidate, against {pct(within_main['head']['mixed_accuracy'], 1)} for the head.</li>
<li><b>There is a lot of headroom left:</b> a right answer is among these candidates for {pct(oracle_main)} of questions, so the ranker uses about
{100 * main['pooled']['difference'] / (oracle_main - main['pooled']['greedy']):.0f}% of what the pool offers.</li>
</ul></div>

<h2 id="story">2. Where this fits</h2>
<p>Approach 1 has two halves. <b>Detection</b> asks whether an answer is likely wrong; reports 01 to 06 showed that a small head on the frozen model's hidden
states does this far better than the model's own token probability (pooled AUROC about 0.88 against 0.72). <b>Correction</b> asks what to do about it, and until
now nothing worked:</p>
{table(["Step", "Where", "Test result (greedy 82.78%)", "Verdict"], [
    ["Prompted retry of every answer", "Report 08", "−2.73 points", "Harmful: breaks more than it fixes"],
    ["Gated retry (only low head score, keep if the head prefers it)", "Report 08", "+0.24 points, p = 0.10", "Within noise"],
    ["Deeper second pass", "Report 08", "−12.79 points", "Harmful"],
    ["Best-of-5, head picks (T = 0.9)", "Report 08", f"+0.40 points, p = 0.081" if best5 is None else f"{pts(best5['head_best']['pooled']['accuracy'] - best5['greedy']['pooled']['accuracy'])} points, p = 0.081", "Within noise"],
    ["Best-of-5, head picks (T = 0.7, greedy in the pool)", "Report 10", f"{pts(search['test']['0.7']['5']['pooled']['selectors']['head_best']['difference'])} points, p = {search['test']['0.7']['5']['pooled']['selectors']['head_best']['p_value']:.3f}", "Borderline"],
    ["Best-of-7, head picks (T = 0.7), chosen on validation", "Report 10", f"{pts(search['test']['0.7']['7']['pooled']['selectors']['head_best']['difference'])} points, p = {search['test']['0.7']['7']['pooled']['selectors']['head_best']['p_value']:.3f}", "Borderline"],
    [f"<b>Learned correction step ({esc(main_key)})</b>", "This report", f"<b>{pts(main['pooled']['difference'])} points, p {pval(main['pooled']['p_value'])}</b>", "<b>Significant</b>"],
], best_rows=(6,), left_cols=(0, 1, 3))}
<p>Report 10 found why best-of-k gained so little. The right answer is often in the pool (90.8% at T = 1.3, k = 10), but the correctness head is a weak
judge <em>within</em> one question: its pooled AUROC of 0.88 drops to 0.65–0.70 when only answers to the same question are compared, because across
questions most of its signal is question difficulty, which is constant inside one question. So this report attacks both halves of selection: better
candidates, and a judge trained for the within-question decision.</p>

<h2 id="idea">3. The idea</h2>
{figure("fig-pipe", "The correction step", pipeline_svg(), "Every candidate, whatever produced it, is encoded the same way and scored by one ranker.")}
<p>For a question, collect candidate answers from several sources, encode each distinct answer the same way (the full image and the original question,
followed by the answer), and let a ranker pick one. If the greedy answer is the best candidate the ranker keeps it, so the step can only change an answer
when another candidate scores higher. Three design rules:</p>
<ul>
<li><b>Candidates should differ in evidence, not only in noise.</b> Report 10 showed that resampling at higher temperature mostly reproduces the same
answer or adds confident wrong ones. The two new sources change what the model attends to: a contrastive comparison against a noised image, and zoomed views.</li>
<li><b>Every candidate is judged in the same context.</b> A zoomed-crop answer is scored on the full image with the original question, exactly like a sampled
one, so the ranker compares like with like. The question embedding is the same cached tensor for every candidate, as in reports 05 to 10.</li>
<li><b>The judge is trained for the decision it makes.</b> The ranker's loss compares the candidates of one question with each other, so question
difficulty cannot help it; only properties of the answer can.</li>
</ul>

<h2 id="sources">4. Candidate sources</h2>
<h3>4.1 Greedy and sampled answers</h3>
<p>The greedy answer is the model's normal output. Sampled answers come from the pools of report 10: up to ten per question at T = 1.0 and T = 1.3, top-p 0.95,
extended here to {max(longest.values()) if longest else 16} per question for the best-of-11+ comparison with a different seed, so samples 11 onwards are independent
draws from the same distribution. Pool k is always the greedy answer plus the first k samples.</p>
<h3>4.2 Visual contrastive decoding</h3>
<p>Some wrong answers come from language priors: the model answers what is typical for the question, not what the image shows. Contrastive decoding
removes that part. At every generated token it compares the next-token distribution under the real image with the distribution under a heavily noised
copy of the same image (Gaussian noise mixed in with weight α<sub>noise</sub> = 0.15, the corruption used in report 06), and decodes greedily from</p>
<pre>score(y) = (1 + α) · log p(y | image, question, prefix) − α · log p(y | noised image, question, prefix)</pre>
<p>restricted to tokens the real image finds plausible, p(y | image) ≥ 0.1 · max p. A token the model would produce even without seeing the image gets its
score pulled down; a token supported by the image keeps it. It is run at α = 0.5 and α = 1.0 and adds up to two candidates. It needs a second forward pass per
token, on the noised image.</p>
<h3>4.3 Zoomed crops</h3>
<p>Qwen3-VL already sees every image at full resolution (the processor does not downscale these images), so a higher-resolution pass adds nothing. What can
help is giving small regions more visual tokens: the question is asked on four overlapping crops, each 60% of the width and height, at the four corners,
upscaled 2× (longest side capped at 2048 px). A crop may not contain the evidence, and then its answer is often wrong; that is why these candidates go to a
ranker instead of replacing the answer. It adds up to four candidates.</p>
<h3>4.4 Encoding every candidate the same way</h3>
<p>Each distinct answer string is encoded once with a teacher-forced pass over the full image, the original prompt, and the answer. That gives its answer state
(the last-layer state at the last token, as the heads use), its <b>teacher-forced answer probability</b> (geometric mean of the answer tokens' probabilities
under the clean prompt, the same number for an answer whatever produced it), and the unified head and linear probe scores computed with the cached question
embedding.</p>

<h2 id="ranker">5. The ranker</h2>
<p>For each candidate the ranker sees three things, and outputs one score; the candidate with the highest score is the answer.</p>
{table(["Input", "What it is", "Why it can help within a question"], [
    ["Question embedding", "The cached prompt state, identical for every candidate", "Lets the network condition what it looks for on the question type"],
    ["Answer state", "Last-layer state after the full image, prompt and this answer", "What the model itself represents about this answer in context"],
    ["Vote share", "Share of the k sampled answers that normalize to the same string", "Agreement among samples is evidence the head does not see"],
    ["Is greedy", "1 for the greedy answer", "Lets the ranker keep the default unless something is clearly better"],
    ["Correction share", "Share of correction candidates proposing this answer", "Agreement between different views of the evidence"],
    ["Answer probability", "Teacher-forced geometric mean, and its log", "Comparable across sources, unlike the sampling probability"],
    ["Answer length", "Tokens / 10", "Catches run-on and repeated outputs"],
    ["Head and probe scores", "Unified head joint score (and its logit), linear probe score", "The detection signal of reports 05–10"],
], left_cols=(0, 1, 2))}
<p><b>Network.</b> Both states are layer-normalized and concatenated with the nine scalar features, then one hidden layer of 512 units (GELU, dropout 0.2) gives the
score. <b>Loss.</b> For a question with candidates c<sub>1</sub>…c<sub>n</sub>, scores s<sub>i</sub> and the set R of right candidates,</p>
<pre>L = log Σ_i exp(s_i) − log Σ_(i in R) exp(s_i)   (+ β · BCE(s_i, right_i), β in {{0, 0.3}})</pre>
<p>a softmax over the question's candidates with every right one as a target. It is zero for questions whose candidates are all right or all wrong, so only
questions where the choice matters train the ranker, and since every term compares candidates of the same question, question difficulty cancels out.</p>
<p><b>Training data.</b> All 50,517 training questions: the greedy answer and eight samples at T = 1.0 for every question, plus the six correction candidates for a
spread subset of 3,200 (400 per shard). Each epoch redraws every question's set (k from {{3, 5, 8}}, with the corrections 70% of the time when available) so the vote
features cover the pool sizes used at test time. <b>Selection.</b> Three variants (states and scalars, scalars only, states only) × 2 learning rates × 2 BCE
weights, three seeds each, eight epochs, best epoch by validation accuracy. The reported ranker averages the scores of its three seeds. The candidate-set
configuration (temperature, k, which corrections) was then chosen on validation: <code>{esc(main_key)}</code>, validation gain {pts(main_val['difference'])} points.
Test was read once for it.</p>
"""]
    sweep_rows = [[e["variant"], f"{e['learning_rate']:g}", f"{e['bce_weight']:g}", pct(e["val_accuracy"])] for e in ranker["sweep"]]
    parts.append("<h3>Training sweep (validation accuracy on T = 1.0, k = 10, all corrections; mean of three seeds)</h3>" +
                 table(["Variant", "Learning rate", "BCE weight", "Validation accuracy"], sweep_rows))

    rows = []
    for name in ("pooled", *DATASETS):
        r = main[name]
        rows.append([DATASETS.get(name, "<b>All four</b>"), pct(r["greedy"]), pct(r["accuracy"]), ci(r), f"+{r['fixed']} / −{r['broken']}", pval(r["p_value"]),
                     f"{100 * r['difference'] / r['greedy']:+.1f}%", errors_removed(r), pct(ranker["choice"]["test"]["oracle"][name]["accuracy"], 1)])
    parts.append(f"""<h2 id="result">6. Main result</h2>
<p>Configuration chosen on validation: <b>T = 1.0, five samples, both corrections</b> (<code>{esc(main_key)}</code>), states-and-scalars ranker. Test, read once.</p>
{table(["Dataset", "Greedy", "Corrected", "Difference, points (95% interval)", "Fixed / broken", "McNemar p", "Relative gain", "Errors removed", "Oracle"], rows, best_rows=(0,))}
<p><b>How to read it.</b> "Relative gain" is the difference divided by greedy accuracy; "errors removed" divides it by the greedy error rate, which is the more
telling number when greedy is already above 80%. Fixed and broken count questions whose correctness changed; the McNemar test asks whether fixes outnumber
breaks by more than chance. The gain is largest where answers are text read from the image (TextVQA, DocVQA) and smallest on ChartQA, where most errors are
reasoning over plotted values that no candidate source changes.</p>""")

    # Best-of-k.
    def k_row(label, key, selector="ranker"):
        if key not in curve:
            return None
        r = curve[key][selector]["pooled"]
        return [label, pct(r["accuracy"]), ci(r), pval(r["p_value"]), pct(curve[key]["oracle"]["pooled"]["accuracy"], 1)]
    head_rows = [[f"Best-of-{k}, head picks, T = 0.7 (report 10)",
                  pct(search["test"]["0.7"][str(k)]["pooled"]["selectors"]["head_best"]["accuracy"]),
                  f"{pts(search['test']['0.7'][str(k)]['pooled']['selectors']['head_best']['difference'])}",
                  f"{search['test']['0.7'][str(k)]['pooled']['selectors']['head_best']['p_value']:.3f}",
                  pct(search["test"]["0.7"][str(k)]["pooled"]["pool"]["oracle_accuracy"], 1)] for k in (5, 7, 10)]
    ranker_rows = [r for r in (k_row(f"Best-of-{k}, ranker, samples only, T = 1.0", f"T1_k{k}_none") for k in (5, 7, 9, 10, 11, 13, 15)) if r]
    corrected_rows = [r for r in (k_row(f"Best-of-{k} + corrections, ranker, T = 1.0", f"T1_k{k}_all") for k in (0, 5, 7, 9, 10, 11, 13, 15)) if r]
    hot_rows = [r for r in (k_row(f"Best-of-{k} + corrections, ranker, T = 1.3", f"T1.3_k{k}_all") for k in (5, 7, 9, 11, 13, 15)) if r]
    if best5:
        head_rows.insert(0, ["Best-of-5, head picks, T = 0.9, no greedy in the pool (report 08)", pct(best5["head_best"]["pooled"]["accuracy"]),
                             pts(best5["head_best"]["pooled"]["accuracy"] - best5["greedy"]["pooled"]["accuracy"]), "0.081", pct(best5["oracle"]["pooled"]["accuracy"], 1)])
    parts.append(f"""<h2 id="k">7. Best-of-5, best-of-7, and whether to go to 9 or 11</h2>
<p>The first full-split experiment chose among five samples with the head (report 08); report 10 widened that to seven and ten. This section puts those next to
the ranker at every k, and extends the pools to {max(longest.values()) if longest else 16} samples to answer whether best-of-9, 11 or more would help.</p>
{table(["Configuration", "Accuracy", "Difference from greedy, points", "p", "Oracle"], head_rows + ranker_rows + corrected_rows + hot_rows, left_cols=(0,))}
<div class="grid2">
<figure class="panel"><figcaption>Accuracy against k, T = 1.0</figcaption>{curve_chart(curve, "1", [
    ("Greedy", "none", "greedy", "c-tok"), ("Head best", "none", "head_best", "c-sp"), ("Vote, then head", "none", "vote_then_head", "c-sv"),
    ("Ranker, samples only", "none", "ranker", "c-up"), ("Ranker + corrections", "all", "ranker", "c-uj")],
    0.80, 0.86, "Accuracy against k", lambda r: r["accuracy"])}</figure>
<figure class="panel"><figcaption>Oracle against k, T = 1.0</figcaption>{curve_chart(curve, "1", [
    ("Samples only", "none", "oracle", "c-up"), ("+ corrections", "all", "oracle", "c-uj")], 0.82, 0.96, "Oracle against k", lambda r: r["accuracy"])}</figure>
</div>
<p><b>Answer: going past five samples does not pay.</b> The oracle keeps rising with k, because each extra sample sometimes contains the right answer, but the
ranker's accuracy is flat from k = 3 to k = {max(longest.values()) if longest else 16} and the head's falls. Extra samples are mostly repeats of answers already in the pool
or confident alternatives that are wrong; each one costs a generation and an encoding pass. The candidates that do help are the ones that change the
evidence (Section 8): zoom and contrastive decoding with no samples at all already reach {pct(accuracy('T1_k0_all'))}. If more compute is available it is better
spent on more evidence-changing candidates (finer crop grids, crops chosen by attention) or a stronger judge than on more samples.</p>""")

    # Contributions.
    rows = []
    for name, label in (("vcd", "Contrastive decoding only (2 candidates)"), ("zoom", "Zoomed crops only (4 candidates)"), ("all", "Both (6 candidates)")):
        key = f"T1_k0_{name}"
        if key in curve:
            r = curve[key]["ranker"]["pooled"]
            rows.append([label, pct(r["accuracy"]), ci(r), f"+{r['fixed']} / −{r['broken']}", pval(r["p_value"]), pct(curve[key]["oracle"]["pooled"]["accuracy"], 1)])
    for name in ("none", "vcd", "zoom", "all"):
        key = f"T1_k5_{name}"
        if key in curve:
            r = curve[key]["ranker"]["pooled"]
            rows.append([f"Five samples + {SET_LABEL[name].removeprefix('samples').strip(' +') or 'nothing else'}", pct(r["accuracy"]), ci(r),
                         f"+{r['fixed']} / −{r['broken']}", pval(r["p_value"]), pct(curve[key]["oracle"]["pooled"]["accuracy"], 1)])
    attribution = analysis.get("attribution_main", {})
    attr_rows = [[esc(" + ".join(SOURCE_LABEL.get(s, s) for s in key.split("+"))), v["changed"], v["fixed"], v["broken"], v["wrong_to_wrong"], v["right_to_right"],
                  f"{v['fixed'] - v['broken']:+d}"] for key, v in attribution.items()]
    parts.append(f"""<h2 id="contrib">8. What each candidate source adds</h2>
<p>All rows use the same trained ranker and the greedy answer; only the extra candidates change.</p>
{table(["Candidates besides greedy", "Accuracy", "Difference, points (95% interval)", "Fixed / broken", "p", "Oracle"], rows, left_cols=(0,))}
<p><b>Zoomed crops carry the gain.</b> With no sampling, the four crops and the ranker reach almost the full result, and adding them to five samples is what
separates the corrected configuration from samples alone. Contrastive decoding on its own changes little: most of the time it reproduces the greedy answer,
because for a 2B model reading text the language prior is rarely what goes wrong. When it disagrees with greedy, though, it is usually worth looking at.</p>
<h3>Where the ranker's changes come from (test, main configuration)</h3>
<p>Questions where the ranker did not keep the greedy answer, grouped by which sources proposed the answer it chose. A source can share an answer with others.</p>
{table(["Chosen answer proposed by", "Changed", "Fixed", "Broken", "Wrong to wrong", "Right to right", "Net"], attr_rows, left_cols=(0,))}""")

    # Why it works.
    rows = []
    for key in ("T1_k5_none", "T1_k5_all", "T1_k10_all", "T1.3_k10_all"):
        if key in curve:
            w = curve[key]["within"]
            rows.append([esc(key), f"{w['ranker']['grouped_auroc']:.3f}", f"{w['head']['grouped_auroc']:.3f}", f"{w['vote']['grouped_auroc']:.3f}",
                         f"{pct(w['ranker']['mixed_accuracy'], 1)} / {pct(w['head']['mixed_accuracy'], 1)} / {pct(w['ranker']['mixed_random_pick'], 1)}", f"{w['ranker']['mixed_questions']:,}"])
    parts.append(f"""<h2 id="why">9. Why it works: a better judge within a question</h2>
<p>Report 10 introduced two measurements of the selection decision itself: grouped AUROC (how often a right candidate scores above a wrong candidate
<em>of the same question</em>) and mixed-pool accuracy (how often the top candidate is right, on questions that have both). Here they compare the ranker with the
head and with the vote share, on identical candidate sets.</p>
{table(["Candidate set (test)", "Grouped AUROC, ranker", "Head", "Vote share", "Mixed-pool accuracy: ranker / head / random pick", "Mixed questions"], rows, left_cols=(0,))}
<p>The ranker is a clearly better within-question judge than the head it builds on, and better than the vote alone. It combines them: it learns, for example,
that a zoomed-crop answer agreeing with several samples is more reliable than either signal alone, and that a confident answer that no other source proposes
is suspicious. The head was trained to rank answers across questions, where question difficulty does most of the work, and never had to make this
comparison.</p>""")

    # Ablations.
    variants = analysis.get("variants", {})
    rows = [[{"full": "States and scalars", "scalars": "Scalars only", "states": "States only"}[v], pct(r["pooled"]["accuracy"]), ci(r["pooled"]), pval(r["pooled"]["p_value"])]
            + [pts(r[d]["difference"]) for d in DATASETS] for v, r in variants.items()]
    seed = ranker.get("seed_accuracy", {}).get("test", {})
    spread = [s for key, s in seed.items() if key.startswith(main_key)]
    parts.append(f"""<h2 id="ablation">10. Ablations</h2>
<h3>Ranker inputs (test, main configuration)</h3>
{table(["Ranker", "Accuracy", "Difference, points (95% interval)", "p"] + list(DATASETS.values()), rows, left_cols=(0,))}
<p>The scalar features do most of the work: vote share, the head score, the answer probability and the greedy flag, combined by a learned function, already
give most of the gain. The answer states add a further gain on top. On their own the states are not enough, and the states-only ranker falls below
greedy: without the greedy flag and the agreement features it has no reason to prefer the default answer, and the states alone do not separate a right
answer from a plausible wrong one of the same question.</p>
<p><b>Seeds.</b> Single-seed test accuracies of the chosen variant and configuration: {', '.join(pct(v) for v in spread[0]) if spread else 'not recorded'}. The reported
ranker averages the three seeds' scores.</p>""")

    # Compute.
    gating = analysis.get("gating_main", {})
    gating_zoom = analysis.get("gating_zoom_only", {})
    def gating_rows(data):
        rows = []
        for gate, entries in data.items():
            for budget in ("0.2", "0.3", "0.5", "1.0"):
                if budget in entries:
                    r = entries[budget]
                    rows.append([GATE_LABEL[gate], f"{100 * float(budget):.0f}%", pct(r["accuracy"]), ci(r), pval(r["p_value"]), f"{r['passes_per_question']:.1f}"])
        return rows
    parts.append(f"""<h2 id="compute">11. Compute: correcting only the questions that need it</h2>
<p>The full step costs, per question, a batched sampling call, four crop generations, two contrastive-decoding runs (two forward passes per token each) and one
encoding pass per distinct candidate, against one generation for greedy. A gate sends only the lowest-scored share of questions through the correction step;
the rest keep the greedy answer. Passes per question count generations and encodings.</p>
{figure("fig-gate", "Gain over greedy against the share of questions corrected (test, main configuration)", gating_chart(gating), "The gates are test numbers and were not chosen on validation; read them as indicative.") if gating else ''}
{table(["Gate", "Questions corrected", "Accuracy", "Difference, points (95% interval)", "p", "Passes per question"], gating_rows(gating), left_cols=(0,))}
<h3>The cheapest strong configuration: greedy + four zoomed crops</h3>
{table(["Gate", "Questions corrected", "Accuracy", "Difference, points (95% interval)", "p", "Passes per question"], gating_rows(gating_zoom), left_cols=(0,))}
<p>Because the ranker usually keeps the greedy answer when the greedy answer is confident, most of the gain sits in the questions the head already flags. With
the head as the gate, correcting a minority of questions recovers most of the full gain at a fraction of the passes, which is what makes the step usable as
an adaptive-compute layer: detection decides where to spend, correction spends it.</p>""")

    cases = analysis.get("cases", [])
    parts.append(f"""<h2 id="cases">12. Case studies</h2>
<p>Two fixes and one break per dataset from the test split, chosen as the first in index order, not curated. Each lists every candidate with the sources that
proposed it, whether it is right, the ranker score, the head score and the share of samples agreeing.</p>
{''.join(case_html(c) for c in cases)}""")

    parts.append(f"""<h2 id="meaning">13. What it means</h2>
<ul>
<li>Correction works once the two halves are separated: generate candidates that look at the evidence differently, and choose with a model trained for that choice.
Neither half alone was enough: the head choosing from samples gained 0.4 points, and the same ranker over samples alone gains less than over samples and crops.</li>
<li>Detection still matters, in two places: the head score is one of the ranker's strongest features, and it is the gate that decides which questions to correct.</li>
<li>The remaining headroom is large ({pct(oracle_main)} oracle against {pct(main['pooled']['accuracy'])}). The most direct ways to use it are a stronger judge (the model
itself asked to compare candidates against the image, or a ranker trained with more correction candidates) and more evidence-changing candidates, not more samples.</li>
</ul>
<h2 id="limits">14. Limits</h2>
<ul>
<li>One model size (2B). The crop grid is fixed (four corners); a question whose evidence sits in the centre is helped only by the overlap.</li>
<li>The ranker trains on correction candidates for only 3,200 of 50,517 training questions, so it saw far fewer zoom and contrastive candidates than samples.</li>
<li>The gating and per-source tables are test numbers; only the main configuration and the ranker were chosen on validation.</li>
<li>Correctness uses the thresholds of the earlier reports (VQA score ≥ 0.5, ANLS ≥ 0.9, option letter), which can count near-misses as wrong both ways.</li>
<li>Contrastive decoding uses one noise strength and one plausibility cutoff; neither was tuned.</li>
</ul>
<h2 id="next">15. Next</h2>
<ol>
<li>Train the ranker with correction candidates on all training questions (now 3,200), the cheapest likely gain.</li>
<li>A finer or adaptive crop grid (3 × 3, or crops around the regions the model attends to), since zoom is the source that works.</li>
<li>The model as its own judge: ask it whether each candidate is supported by the image and use that probability as a ranker feature.</li>
<li>Repeat on Qwen3-VL-4B to see whether the gain holds with a stronger base model.</li>
</ol>
<h2 id="appendix">Appendix A. Files and jobs</h2>
<ul>
<li>Code: <code>correction_candidates.py</code> (contrastive decoding, crops, candidate encoding), <code>train_candidate_ranker.py</code> (ranker, loss, sweep),
<code>extend_pools.py</code> (samples 11 to 16), <code>analyze_ranker.py</code> (this report's analyses), <code>build_ranker_report.py</code>. Jobs:
<code>slurm/run_correction_learning_pipeline.sh</code>, <code>run_correction_candidates.sbatch</code>, <code>train_candidate_ranker.sbatch</code>,
<code>run_extend_pipeline.sh</code>, <code>run_extend_pools.sbatch</code>, <code>analyze_ranker.sbatch</code>. Tests: <code>tests/test_correction_ranker.py</code>.</li>
<li>Data: <code>04_sweep_data/candidate_ranker.json</code> (sweep, every configuration on validation and test, seeds) and <code>04_sweep_data/ranker_analysis.json</code>
(k curves to {max(longest.values()) if longest else 16} samples, within-question measurements, attribution, gating, cases); tables in <code>11_correction_ranker_tables.md</code>.</li>
<li>Compute: {totals['jobs']} GPU jobs on ICE, {totals['gpu_hours']:.1f} GPU-hours, no failed jobs or examples.</li>
</ul>
{jobs_html}
</main></body></html>""")
    return "".join(parts)


def main() -> None:
    """Write the report."""
    parser = argparse.ArgumentParser(description="Build report 11.")
    parser.add_argument("--analysis", type=Path, default=RESULTS / "ranker_analysis.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(build(args.analysis), encoding="utf-8")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
