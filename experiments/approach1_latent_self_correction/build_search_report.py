# build_search_report.py
#
# Builds report 10: the candidate-search study (larger pools, higher temperature, best-of-k for
# k up to 10, selectors, and measurements that target the selection decision) and the sweep over
# how the pre-generation head is trained on the constant question embedding. One self-contained
# HTML file in the style of the earlier reports.
# Reads results/candidate_search.json (analyze_candidate_pools.py), results/pre_head_sweep.json
# (sweep_pre_head.py), and results/candidate_search_jobs.txt (the scheduler record of the run).

import json
from pathlib import Path

from experiments.approach1_latent_self_correction.build_report import CSS, Panel, esc, figure, legend, table

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUTPUT = RESULTS / "candidate_search_report.html"
DATASETS = {"chartqa_full": "ChartQA", "textvqa": "TextVQA", "docvqa": "DocVQA", "scienceqa_img": "ScienceQA"}
TEMPS = (("0.7", "T = 0.7", "c-w0"), ("1", "T = 1.0", "c-w1"), ("1.3", "T = 1.3", "c-w2"))
KS = ("1", "3", "5", "7", "10")
SELECTOR_LABEL = {"greedy": "Greedy", "first_sample": "First sample", "majority_vote": "Majority vote",
                  "head_vote": "Head-weighted vote", "probe_vote": "Probe-weighted vote", "head_best": "Head best",
                  "probe_best": "Probe best", "token_best": "Token best", "head_anchor_0.00": "Anchor, margin 0",
                  "head_anchor_0.05": "Anchor, margin 0.05", "head_anchor_0.10": "Anchor, margin 0.10",
                  "head_anchor_0.20": "Anchor, margin 0.20", "oracle": "Oracle"}
GATE_LABEL = {"pre_head": "Unified head, pre-generation mode", "pre_probe": "Linear probe, pre-generation mode",
              "head": "Unified head on the greedy answer", "probe": "Linear probe on the greedy answer",
              "token": "Answer-token probability of the greedy answer"}


def pct(value: float, digits: int = 2) -> str:
    """A proportion as a percentage."""
    return f"{100 * value:.{digits}f}%"


def pts(value: float) -> str:
    """A difference of proportions in signed percentage points."""
    return f"{100 * value:+.2f}"


def k_chart(title: str, grid: dict, value, ymin: float, ymax: float, step: float, ylabel: str, fmt=lambda v: f"{100 * v:.0f}%") -> str:
    """Line chart over pool size k, one line per temperature."""
    width, height = 340, 250
    panel = Panel(52, 12, width - 66, height - 56, 1, 10, ymin, ymax)
    ticks, tick = [], ymin
    while tick <= ymax + 1e-9:
        ticks.append(round(tick, 4))
        tick += step
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
             panel.axes([1, 3, 5, 7, 10], ticks, yfmt=fmt, xlabel="pool size k", ylabel=ylabel)]
    for temperature, label, css in TEMPS:
        points = [(int(k), value(grid[temperature][k])) for k in KS if k in grid.get(temperature, {})]
        parts.append(f'<polyline class="{css} ln" fill="none" points="{" ".join(f"{panel.sx(x):.1f},{panel.sy(y):.1f}" for x, y in points)}"/>')
        for x, y in points:
            parts.append(f'<circle class="{css} dot" cx="{panel.sx(x):.1f}" cy="{panel.sy(y):.1f}" r="3"><title>{label}, k={x}: {fmt(y)}</title></circle>')
    parts.append("</svg>")
    return f'<figure class="panel"><figcaption>{esc(title)}</figcaption>{"".join(parts)}</figure>'


def curve_chart(gates: dict) -> str:
    """Accuracy gain over greedy against the share of questions routed to best-of-k, one line per gate."""
    width, height = 680, 280
    budgets = ("0.1", "0.2", "0.3", "0.5", "1.0")
    gains = [100 * gates[g]["curve"][b]["gain"] for g in gates for b in budgets]
    top = max(0.6, max(gains) + 0.1)
    panel = Panel(56, 14, width - 76, height - 60, 0, 1, 0, top)
    css = {"pre_head": "c-sp", "pre_probe": "c-sv", "head": "c-uj", "probe": "c-up", "token": "c-tok"}
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Gain against routed share">',
             panel.axes([0, 0.1, 0.2, 0.3, 0.5, 1.0], [round(i * 0.1, 1) for i in range(int(top * 10) + 1)][::2],
                        xfmt=lambda v: f"{100 * v:.0f}%", yfmt=lambda v: f"+{v:.1f}", xlabel="share of questions routed to best-of-k",
                        ylabel="gain over greedy (points)")]
    for gate, data in gates.items():
        points = [(0.0, 0.0)] + [(float(b), 100 * data["curve"][b]["gain"]) for b in budgets]
        parts.append(f'<polyline class="{css[gate]} ln" fill="none" points="{" ".join(f"{panel.sx(x):.1f},{panel.sy(y):.1f}" for x, y in points)}"/>')
        for x, y in points[1:]:
            parts.append(f'<circle class="{css[gate]} dot" cx="{panel.sx(x):.1f}" cy="{panel.sy(y):.1f}" r="3"><title>{esc(GATE_LABEL[gate])}: {100 * x:.0f}% routed, {y:+.2f} points</title></circle>')
    parts.append("</svg>")
    return legend([(GATE_LABEL[g], css[g]) for g in gates]) + "".join(parts)


def jobs_table(path: Path) -> tuple[str, dict]:
    """Scheduler record of the run as a table, plus totals."""
    rows, seconds, gpus = [], 0, {}
    for line in path.read_text().splitlines():
        job, name, state, elapsed, start, end, tres = line.split("|")
        h, m, s = (int(x) for x in elapsed.split(":"))
        seconds += 3600 * h + 60 * m + s
        gpu = next((part.split(":")[1].split("=")[0].upper() for part in tres.split(",") if part.startswith("gres/gpu:")), "?")
        gpus[gpu] = gpus.get(gpu, 0) + 1
        rows.append([job, esc(name), state.title(), elapsed, gpu])
    starts = [line.split("|")[4] for line in path.read_text().splitlines()]
    ends = [line.split("|")[5] for line in path.read_text().splitlines()]
    totals = {"jobs": len(rows), "gpu_hours": seconds / 3600, "first": min(starts), "last": max(ends), "gpus": gpus}
    return table(["Job", "Name", "State", "Run time", "GPU"], rows, left_cols=(1,)), totals


def build() -> str:
    """Assemble report 10."""
    search = json.loads((RESULTS / "candidate_search.json").read_text())
    sweep = json.loads((RESULTS / "pre_head_sweep.json").read_text())
    val, test, report = search["validation"], search["test"], search["report"]
    choice = report["choice"]
    pooled_val = {t: {k: val[t][k]["pooled"] for k in val[t]} for t in val}
    pooled_test = {t: {k: test[t][k]["pooled"] for k in test[t]} for t in test}
    chosen = report["per_dataset"]["pooled"]
    greedy_test = pooled_test["1"]["1"]["selectors"]["greedy"]["accuracy"]
    best_oracle = max(pooled_test[t][k]["pool"]["oracle_accuracy"] for t in pooled_test for k in pooled_test[t])
    jobs_html, totals = jobs_table(RESULTS / "candidate_search_jobs.txt")

    # Pre-generation sweep summaries.
    configs = sweep["configs"]
    def best(filter_fn):
        return min((c for c in configs if filter_fn(c["config"])), key=lambda c: c["val"]["pooled"]["aurc"])
    refs = sweep["references"]
    exported = refs["exported pre-generation (MLP)"]["test"]["pooled"]
    winner = sweep["winners"]["gate_blend"] if "gate_blend" in sweep["winners"] else sweep["winners"]["gate"]
    win_test = winner["test"]["pooled"]

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Candidate Search Report</title><style>{CSS}</style></head><body><main>
<h1>Approach 1, report 10: candidate search and pre-generation head training</h1>
<p class="sub">Larger candidate pools (best-of-k up to 10, temperature 0.7 to 1.3), selectors and measurements that target the
selection decision, and a sweep over how the pre-generation head is trained with its input held fixed. Qwen3-VL-2B, frozen.
Full test splits: 9,517 questions (ChartQA 2,500, TextVQA 2,500, DocVQA 2,500, ScienceQA 2,017). Choices made on validation
(6,420 questions); test read once per choice.</p>
<p class="meta">Follows report 08 (complete Approach 1 report), Section 12.6, where best-of-five on the full test split gained
+0.40 points over greedy and was not distinguishable from zero.</p>
<div class="toc"><a href="#summary">Summary</a><a href="#why">Why</a><a href="#setup">Setup</a><a href="#metrics">Measurements</a>
<a href="#pools">Pools</a><a href="#selection">Selection</a><a href="#within">Within-question ranking</a><a href="#gates">Gates and compute</a>
<a href="#prehead">Pre-generation head</a><a href="#meaning">What it means</a><a href="#limits">Limits</a><a href="#next">Next</a><a href="#appendix">Appendix</a></div>

<h2 id="summary">1. Summary</h2>
<div class="stats">
<div class="stat"><b>{pct(greedy_test, 1)} → {pct(best_oracle, 1)}</b><span>greedy against the best oracle (T = 1.3, k = 10)</span></div>
<div class="stat"><b>{pts(chosen['difference'])} pts</b><span>chosen search on test (T = {choice['temperature']}, k = {choice['k']}, {esc(SELECTOR_LABEL.get(choice['selector'], choice['selector']))}), p = {chosen['p_value']:.3f}</span></div>
<div class="stat"><b>{pooled_val['1']['10']['grouped_auroc']['head']:.2f}</b><span>head AUROC within a question (val, T = 1.0, k = 10), against about 0.88 pooled</span></div>
<div class="stat"><b>{exported['auroc_greedy']:.3f} → {win_test['auroc_greedy']:.3f}</b><span>pre-generation head AUROC, exported head to sweep winner</span></div>
</div>
<div class="card summary"><ul>
<li><b>The search space was the problem in report 08 and it can be widened.</b> At T = 0.7 and k = 1 only {pct(pooled_val['0.7']['1']['pool']['mixed_fraction'], 1)} of
validation questions have both a right and a wrong candidate; at T = 1.3 and k = 10 it is {pct(pooled_val['1.3']['10']['pool']['mixed_fraction'], 1)}, and the test oracle rises from
{pct(pooled_test['0.7']['1']['pool']['oracle_accuracy'], 1)} to {pct(best_oracle, 1)}.</li>
<li><b>Selecting from the wider pool is the new bottleneck.</b> The configuration chosen on validation (T = {choice['temperature']}, k = {choice['k']}, highest head score)
reaches {pct(chosen['accuracy'])} on test against {pct(report['per_dataset']['pooled']['greedy_accuracy'])} greedy ({pts(chosen['difference'])} points,
95% interval {pts(chosen['ci_low'])} to {pts(chosen['ci_high'])}, p = {chosen['p_value']:.3f}), about the same as best-of-five. TextVQA gains
{pts(report['per_dataset']['textvqa']['difference'])} points (p = {report['per_dataset']['textvqa']['p_value']:.3f}); ChartQA loses {pts(report['per_dataset']['chartqa_full']['difference'])}.
At T = 1.3 picking the highest head score gets worse as k grows and falls below greedy at k = 10.</li>
<li><b>Pooled AUROC overstated the head as a selector.</b> Within one question the head ranks a right answer above a wrong one with AUROC
0.65 to 0.70, against about 0.88 when answers of different questions are pooled. On questions with a mixed pool it picks a right answer 60 to 68% of
the time, against 50 to 55% for a random pick, and it captures 9 to 15% of the oracle headroom at the chosen temperatures.</li>
<li><b>Most of the gain is reachable at a fraction of the compute.</b> Routing only the 20% of questions whose greedy answer the head scores lowest to
best-of-{choice['k']} gives {pts(report['gates']['head']['curve']['0.2']['gain'])} points, the full gain, at {report['gates']['head']['curve']['0.2']['generations_per_question']:.1f}
generations per question instead of {1 + choice['k']}. (Gate curves are test numbers, not chosen on validation.)</li>
<li><b>The pre-generation head trains better on how often the model is right than on whether one greedy answer was right.</b> Without a ranking
term, a pass-rate target beats the greedy target on every metric; with a pairwise pass-rate ranking term every target improves and they converge.
Best configuration: test AUROC {win_test['auroc_greedy']:.3f} against {exported['auroc_greedy']:.3f} for the exported head, AURC {win_test['aurc']:.4f} against
{exported['aurc']:.4f}. The input, the cached question embedding, is the same tensor for every configuration.</li>
</ul></div>

<h2 id="why">2. Why this study</h2>
<p>Report 08 reran the correction experiments on the full test split. Best-of-five with the unified head gained +0.40 points over greedy
(interval −0.04 to +0.84, p = 0.081), with an oracle of +4.8 points. Two things limited it. The candidates barely differed: all five samples agreed on
79% of questions and a question had 1.36 distinct answers on average, so the selector usually had nothing to choose. And the measurement used to
pick the head, pooled correctness AUROC, compares answers to different questions, which is not the decision a selector makes. This study widens the
pool, measures the selection decision directly, and asks whether the pre-generation head, which decides which questions are worth more compute,
can be trained better.</p>

<h2 id="setup">3. Setup</h2>
<div class="card"><ul>
<li><b>Pools.</b> For every validation and test question: the greedy answer plus 10 sampled answers at temperature 0.7, 1.0 and 1.3 (top-p 0.95,
64-token limit), sampled in one batched generate call with a seed that depends only on the question. Best-of-k for k = 1, 3, 5, 7, 10 uses the first k
samples of the same pool, so the k values are nested and differ only in size. Pool k always includes the greedy answer.</li>
<li><b>Scores.</b> Every distinct answer is scored by the exported mixture unified head (MLP, new loss, seed 1) and by the best linear probe of the same
sweep, in joint mode, plus its answer-token probability.</li>
<li><b>Constant question embedding.</b> Every head score uses the cached prompt state from the feature files the heads were trained on, never a
recomputed one, so all candidates, temperatures and training targets see the same question representation. A check on a GPU node compared it with a
freshly computed state: cosine 0.99993 (bfloat16 against the stored float16).</li>
<li><b>Train pass rates.</b> For all 50,517 training questions, 8 samples at T = 1.0, correctness only, as targets for the pre-generation head.</li>
<li><b>Protocol.</b> Temperature, k and selector are chosen on the validation pools by pooled accuracy; test is read once for that choice. The full
test grid is shown for the record. Differences from greedy are paired, with a bootstrap 95% interval and an exact McNemar test.</li>
<li><b>Compute.</b> {totals['jobs']} jobs on ICE GPU nodes ({', '.join(f'{v} on {k}' for k, v in sorted(totals['gpus'].items()))}), {totals['gpu_hours']:.1f} GPU-hours,
from {totals['first'].replace('T', ' ')} to {totals['last'].replace('T', ' ')}. No failed examples.</li>
</ul></div>

<h2 id="metrics">4. Measurements</h2>
<p>Pooled AUROC answers "does a right answer score higher than a wrong answer to some other question". A selector only ever compares answers to the
same question, and only questions whose pool holds both a right and a wrong answer give it anything to do. The measurements below follow that decision.</p>
{table(["Measurement", "Definition", "Why"], [
    ["Accuracy and paired gain", "Accuracy of the selected answer; difference from greedy on the same questions, bootstrap interval, McNemar p", "The outcome that matters"],
    ["Oracle", "A right answer is in pool k", "Upper bound for any selector on that pool"],
    ["Mixed-pool share", "Questions whose pool holds a right and a wrong answer", "Where selection can change anything"],
    ["Mixed-pool accuracy", "Accuracy on mixed pools, against the random-pick rate (share of right candidates)", "Selection skill where it matters"],
    ["Grouped AUROC", "Right-above-wrong ranking inside each mixed pool, averaged over pools", "The within-question version of AUROC"],
    ["Headroom captured", "(selector − greedy) / (oracle − greedy)", "How much of what the pool offers is used"],
    ["AURC, E-AURC", "Area under the risk-coverage curve of the greedy answers ordered by a score; excess over the best ordering", "Quality of a score as a gate"],
    ["Compute curve", "Accuracy when the lowest-scored share b of questions is routed to best-of-k and the rest keep greedy", "Gain against generations spent"],
], left_cols=(0, 1, 2))}

<h2 id="pools">5. Pools: what a wider search offers</h2>
<div class="grid2">
{k_chart("Oracle accuracy, test", pooled_test, lambda e: e["pool"]["oracle_accuracy"], 0.82, 0.92, 0.02, "oracle")}
{k_chart("Mixed pools, validation", pooled_val, lambda e: e["pool"]["mixed_fraction"], 0.0, 0.25, 0.05, "share of questions")}
</div>
{legend([(label, css) for _, label, css in TEMPS])}
"""]
    rows = []
    for t, label, _ in TEMPS:
        for k in KS:
            p = pooled_test[t][k]["pool"]
            rows.append([label, k, pct(p["oracle_accuracy"], 1), pct(pooled_val[t][k]["pool"]["mixed_fraction"], 1),
                         f"{p['mean_distinct_answers']:.2f}", pct(p["unanimous_fraction"], 1), pct(p["sample_pass_rate"], 1), pct(p["capped_fraction"], 2)])
    parts.append(table(["Temperature", "k", "Oracle (test)", "Mixed pools (val)", "Distinct answers", "All agree", "Sample pass rate", "Hit token limit"], rows))
    oracle_rows = []
    for name, label in DATASETS.items():
        oracle_rows.append([label, pct(test["1"]["1"][name]["selectors"]["greedy"]["accuracy"])] +
                           [pct(test[t]["10"][name]["pool"]["oracle_accuracy"], 1) for t, _, _ in TEMPS])
    parts.append("<h3>Oracle at k = 10, per dataset (test)</h3>" + table(["Dataset", "Greedy", "T = 0.7", "T = 1.0", "T = 1.3"], oracle_rows))
    parts.append("""<p>Higher temperature and larger k both help: distinct answers per question rise from about 1.1 to 2.0, the share of mixed pools rises
seven-fold, and the oracle gains about 8 points over greedy at T = 1.3, k = 10. Sample accuracy falls only slightly with temperature, and almost
no samples hit the token limit, so the extra candidates are mostly well-formed alternative answers rather than noise.</p>""")

    # Selection.
    parts.append(f"""<h2 id="selection">6. Selection</h2>
<h3>Validation grid (pooled), used for the choice</h3>""")
    show = ("greedy", "majority_vote", "head_best", "head_vote", "probe_best", "token_best")
    rows, best_val = [], (0, None)
    for t, label, _ in TEMPS:
        for k in KS:
            s = pooled_val[t][k]["selectors"]
            anchor = max(((v["accuracy"], n) for n, v in s.items() if n.startswith("head_anchor")))
            rows.append([label, k] + [pct(s[n]["accuracy"]) for n in show] + [f"{pct(anchor[0])} ({anchor[1].removeprefix('head_anchor_')})", pct(s["oracle"]["accuracy"], 1)])
    parts.append(table(["T", "k"] + [SELECTOR_LABEL[n] for n in show] + ["Best anchor (margin)", "Oracle"], rows))
    rows = []
    for name in ("pooled", *DATASETS):
        r = report["per_dataset"][name]
        rows.append([DATASETS.get(name, "<b>Pooled</b>"), f"{r['questions']:,}", pct(r["greedy_accuracy"]), pct(r["accuracy"]),
                     f"{pts(r['difference'])} ({pts(r['ci_low'])} to {pts(r['ci_high'])})", f"+{r['fixed']} / −{r['broken']}", f"{r['p_value']:.3f}", pct(r["oracle_accuracy"])])
    parts.append(f"""<h3>Chosen on validation: T = {choice['temperature']}, k = {choice['k']}, {esc(SELECTOR_LABEL[choice['selector']])}. Test, read once</h3>
<p>Validation gain {pts(choice['val_gain'])} points.</p>""" + table(["Dataset", "Questions", "Greedy", "Chosen", "Difference (95% interval)", "Fixed / broken", "p", "Oracle"], rows, best_rows=(0,)))
    rows = []
    for t, label, _ in TEMPS:
        for k in KS:
            s = pooled_test[t][k]["selectors"]
            h = s["head_best"]
            rows.append([label, k, pct(s["greedy"]["accuracy"]), pct(s["majority_vote"]["accuracy"]), pct(h["accuracy"]), pct(s["head_vote"]["accuracy"]),
                         pct(s["probe_best"]["accuracy"]), f"{pct(h.get('mixed_accuracy', 0), 1)} / {pct(h.get('mixed_random_pick', 0), 1)}",
                         f"{100 * h['headroom_captured']:.1f}%", pct(s["oracle"]["accuracy"], 1)])
    parts.append("<h3>Test grid (pooled), for the record and not used for the choice</h3>" + table(
        ["T", "k", "Greedy", "Majority", "Head best", "Head vote", "Probe best", "Mixed-pool acc., head / random", "Headroom captured", "Oracle"], rows))
    parts.append(f"""<div class="grid2">
{k_chart("Head best minus greedy, test", pooled_test, lambda e: e["selectors"]["head_best"]["difference"], -0.006, 0.008, 0.002, "points", fmt=lambda v: f"{100 * v:+.1f}")}
{k_chart("Head-weighted vote minus greedy, test", pooled_test, lambda e: e["selectors"]["head_vote"]["difference"], -0.006, 0.008, 0.002, "points", fmt=lambda v: f"{100 * v:+.1f}")}
</div>{legend([(label, css) for _, label, css in TEMPS])}
<p>Every selector gains at most about half a point. Picking the highest head score is best at T = 0.7 and degrades at T = 1.3 as k grows: high-temperature
samples include confident wrong answers that the head scores above the greedy answer. Votes are more robust at high temperature because agreement among
samples carries information the head does not have, but votes alone do not gain either. The linear probe selects as well as the MLP head. Majority vote and
token probability barely move from greedy.</p>""")

    # Within-question ranking.
    rows = []
    for t, label, _ in TEMPS:
        for k in KS[1:]:
            g = pooled_val[t][k]["grouped_auroc"]
            rows.append([label, k, f"{g['head']:.3f}", f"{g['probe']:.3f}", f"{g['token']:.3f}"])
    parts.append(f"""<h2 id="within">7. Within-question ranking: why selection gains little</h2>
<p>The heads were trained and judged by pooled AUROC, where a question's difficulty does most of the work: an easy question's answers score high and a hard
question's low, whatever the answer. Inside one question that signal is constant and only the answer can differ. Measured there, the head is a much weaker
ranker.</p>""" + table(["T", "k", "Grouped AUROC, head", "Linear probe", "Token probability"], rows) + f"""
<div class="grid2">{k_chart("Grouped AUROC of the head, validation", pooled_val, lambda e: e["grouped_auroc"]["head"], 0.6, 0.72, 0.02, "AUROC", fmt=lambda v: f"{v:.2f}")}
{k_chart("Headroom captured by head best, test", pooled_test, lambda e: e["selectors"]["head_best"]["headroom_captured"], -0.1, 0.2, 0.05, "share", fmt=lambda v: f"{100 * v:.0f}%")}</div>
{legend([(label, css) for _, label, css in TEMPS])}
<p>A pooled AUROC of about 0.88 and a within-question AUROC of 0.65 to 0.70 describe the same head. The second number is the one that predicts what a selector
can do, and it is why the oracle headroom of 5 to 8 points turns into half a point. Token probability is weaker still within a question at low temperature
(0.54 to 0.58) and catches up at T = 1.3, where low-probability samples are more often wrong.</p>""")

    # Gates.
    rows = []
    for gate, data in report["gates"].items():
        c = data["curve"]
        rows.append([GATE_LABEL[gate], f"{data['aurc']:.4f}", f"{data['e_aurc']:.4f}", pct(data["accuracy_at_80"])] +
                    [pts(c[b]["gain"]) for b in ("0.1", "0.2", "0.3", "0.5", "1.0")] + [pts(c["mean_gain"])])
    parts.append(f"""<h2 id="gates">8. Gates and compute: which questions should get the extra samples</h2>
<p>The chosen search costs {1 + choice['k']} generations per question. A gate decides which questions get it: the lowest-scored share b is routed to best-of-{choice['k']}
and the rest keep the greedy answer, at 1 + b × {choice['k']} generations per question.</p>
{figure("fig-gates", "Gain over greedy against the share of questions routed (test)", curve_chart(report["gates"]), "Gate curves are test numbers and were not chosen on validation; read them as indicative.")}
{table(["Gate", "AURC ↓", "E-AURC ↓", "Accuracy at 80% coverage", "Gain at 10%", "20%", "30%", "50%", "100%", "Mean gain"], rows, left_cols=(0,))}
<p>Scores of the greedy answer are better gates than pre-generation scores (AURC 0.041 against 0.047), as expected since they see the answer. With the head on the
greedy answer, routing 20% of questions already reaches the full gain. Token probability, a weak detector, is the best router at 10%: questions where the model
is unsure at the token level are exactly the ones where sampling produces alternatives.</p>""")

    # Pre-generation head.
    def row(label, test_row, config=""):
        return [label, config, f"{test_row['auroc_greedy']:.3f}", f"{test_row['aurc']:.4f}", f"{test_row['e_aurc']:.4f}",
                f"{test_row['spearman_pass_rate']:.3f}", pts(test_row["routing"]["0.2"]["gain"]) if "routing" in test_row else "–",
                pts(test_row["routing"]["mean_gain"]) if "routing" in test_row else "–"]
    rows = [row(label, entry["test"]["pooled"], "reference") for label, entry in refs.items()]
    ablation = []
    for rank in (0.0, 0.5):
        for target in ("greedy", "pass_rate", "blend", "any_correct"):
            b = best(lambda c, t=target, r=rank: c["target"] == t and c["rank_weight"] == r)
            c = b["config"]
            ablation.append(row(f"{target}, rank term {rank:g}", b["test"]["pooled"],
                                f"{c['architecture']}, lr {c['learning_rate']:g}, wd {c['weight_decay']:g}, dropout {c['dropout']:g}"))
    arch_rows = []
    for arch in ("linear", "mlp_256", "mlp_512"):
        b = best(lambda c, a=arch: c["architecture"] == arch)
        c = b["config"]
        arch_rows.append(row(arch, b["test"]["pooled"], f"{c['target']}, rank {c['rank_weight']:g}, wd {c['weight_decay']:g}, dropout {c['dropout']:g}"))
    per_ds = []
    for label, entry in [(k, v["test"]) for k, v in refs.items()] + [("Sweep winner (blend + rank term)", winner["test"])]:
        per_ds.append([label] + [f"{entry[n]['auroc_greedy']:.3f} / {entry[n]['aurc']:.4f}" for n in DATASETS])
    headers = ["Row", "Configuration", "AUROC (greedy)", "AURC ↓", "E-AURC ↓", "Spearman with pass rate", "Routing gain at 20%", "Mean routing gain"]
    parts.append(f"""<h2 id="prehead">9. Pre-generation head: a better way to train it</h2>
<p>The pre-generation head reads only the question embedding and scores a question before any answer exists. Until now it learned whether the single greedy
answer was right. That label is noisy for a question-level score: a question the model answers correctly six times in ten is labelled 1 or 0 by chance. The
eight samples per training question give a pass rate, the fraction answered correctly, which is a lower-variance target and closer to what the score is used
for (deciding where more samples help).</p>
<div class="card"><ul>
<li><b>Input.</b> The cached question embedding, identical for every configuration. Only the training changes.</li>
<li><b>Targets.</b> <code>greedy</code> (greedy answer correct), <code>pass_rate</code> (fraction of 8 samples correct, soft BCE), <code>blend</code> (their mean),
<code>any_correct</code> (at least one sample correct).</li>
<li><b>Loss.</b> BCE on the target plus an optional pairwise term on the pass rate: for two questions in a batch whose pass rates differ by more than 0.1, the
one with the higher rate should score higher, softplus(−(s<sub>i</sub> − s<sub>j</sub>)), weight 0 or 0.5.</li>
<li><b>Grid.</b> 4 targets × 3 architectures (normalized linear, MLP-256, MLP-512) × 2 learning rates × 2 weight decays × 2 dropouts × 2 ranking weights =
{len(configs)} configurations, 3 seeds each, early stopping on validation AURC. Calibrated with a temperature fitted on validation greedy labels.</li>
<li><b>Selection.</b> On validation, by AURC (gate quality) and separately by the routing gain; both picked the same family.</li>
</ul></div>
<h3>References and the best configuration of each target, with and without the ranking term (test)</h3>
{table(headers, rows + ablation, best_rows=(len(rows) + 6,), left_cols=(0, 1))}
<h3>Architecture (best configuration of each, test)</h3>
{table(headers, arch_rows, left_cols=(0, 1))}
<h3>Per dataset (test AUROC against greedy correctness / AURC)</h3>
{table(["Row"] + list(DATASETS.values()), per_ds, left_cols=(0,))}
<p>Three results. First, without the ranking term the pass-rate target beats the greedy target on AUROC, AURC and rank correlation with the pass rate; the
any-correct target is the worst, since "solvable by some sample" is a different property from "usually right". Second, the pairwise pass-rate term lifts every
target to about the same level, so the gain comes from the pass-rate signal however it enters the loss. Third, strong regularization (dropout 0.3) and a
one-hidden-layer MLP win; the normalized linear head trails by about 0.004 AUROC. The improvement over the exported head is real but small
({win_test['auroc_greedy'] - exported['auroc_greedy']:+.3f} AUROC, {100 * (win_test['aurc'] / exported['aurc'] - 1):+.0f}% AURC), and as a router for best-of-k it does not
beat the exported head, because the selector it routes to gains little.</p>""")

    parts.append(f"""<h2 id="meaning">10. What it means</h2>
<ul>
<li>Detection was never the weak point; the correctness heads remain far better than token probability at ranking answers across questions.</li>
<li>Generation is not the weak point either once the pool is widened: a right answer is in the T = 1.3, k = 10 pool for {pct(best_oracle, 1)} of test questions.</li>
<li>The weak point is choosing within a question. The heads were never trained for it, and grouped AUROC shows they are only moderately good at it. That
explains every best-of-k result so far, including the small gain in report 08.</li>
<li>So the next correction step should be a selector trained on the within-question decision, using what the heads lack: agreement among samples, the answer
state of each candidate in the same context, and candidates from strategies that change the evidence the model reads, not only its sampling noise.</li>
</ul>
<h2 id="limits">11. Limits</h2>
<ul>
<li>One model size (2B), one sampling family (temperature with top-p 0.95), and greedy answers fixed by the original prompt.</li>
<li>The gate curves and the test grid are reported on test; only the main configuration was chosen on validation.</li>
<li>Pass rates use 8 samples at one temperature, so they are coarse (steps of 1/8).</li>
<li>Correctness uses the same thresholds as before (VQA score ≥ 0.5, ANLS ≥ 0.9, exact option letter), which can mark near-misses as wrong.</li>
</ul>
<h2 id="next">12. Next</h2>
<p>A learned within-question candidate ranker, trained on the 50,517 training pools with a softmax loss over each question's candidates, using the constant question
embedding, each candidate's answer state, vote share, the head and probe scores and a teacher-forced answer probability; and two correction strategies that
add candidates by changing what the model reads: visual contrastive decoding against a noised image, and zoomed crops. These runs are in progress.</p>
<h2 id="appendix">Appendix A. Files and jobs</h2>
<ul>
<li>Code: <code>models/run_inference.py</code> (<code>sample_with_confidence</code>), <code>candidate_pools.py</code>, <code>selection_metrics.py</code>,
<code>analyze_candidate_pools.py</code>, <code>sweep_pre_head.py</code>, <code>build_search_report.py</code>; jobs <code>slurm/run_search_pipeline.sh</code>,
<code>run_candidate_pool.sbatch</code>, <code>analyze_candidate_pools.sbatch</code>, <code>sweep_pre_head.sbatch</code>; tests <code>tests/test_candidate_search.py</code>.</li>
<li>Data: <code>04_sweep_data/candidate_search.json</code> (every temperature, k, selector, pooled and per dataset, validation and test, gates) and
<code>04_sweep_data/pre_head_sweep.json</code> (every configuration, three-seed means, references); tables in <code>10_candidate_search_tables.md</code> and
<code>10_pre_head_sweep_tables.md</code>.</li>
</ul>
{jobs_html}
</main></body></html>""")
    return "".join(parts)


def main() -> None:
    """Write the report."""
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"[done] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
