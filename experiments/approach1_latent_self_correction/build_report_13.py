# build_report_13.py
#
# Builds report 13: larger models and enlarged images as fixes for the 2B model's errors, used on their own
# (routing, a 2B -> 4B -> 8B cascade) and as candidates of the v3 ranker (8B answer, 8B reasoning); the
# ranker trained without the judge; which ranker inputs matter; training-free baselines at matched cost;
# leave-one-dataset-out training; measured latency. One self-contained HTML file in the style of report 12.
# Every number is read from a result file; nothing is typed in by hand.
#
# Reads results/approach1/04_sweep_data/ (published copies of the experiment results):
#   escalation.json                      routing study (analyze_escalation.py)
#   cascade.json                         2B -> 4B -> 8B against 2B -> 8B (analyze_cascade.py)
#   baselines.json                       training-free baselines (compute_baselines.py)
#   latency.json                         measured seconds per component (measure_latency.py)
#   candidate_ranker_v3_multi*.json      rankers v3: with sampled reasoning, input ablations (_drop_*),
#                                        with the escalation candidates (_esc), with 8B reasoning (_esc_cot8b),
#                                        leave-one-dataset-out (_holdout_*)
#   gating_v3_multi*.json                gating of the saved rankers (gate_ranker_v3.py)
# Writes results/approach1/13_escalation_report.html.

import argparse
import json
from pathlib import Path

from experiments.approach1_latent_self_correction.build_report import CSS, figure, table

HERE = Path(__file__).parent
REPO = HERE.parents[1]
DATA = REPO / "results" / "approach1" / "04_sweep_data"
OUTPUT = REPO / "results" / "approach1" / "13_escalation_report.html"
DATASETS = {"chartqa_full": "ChartQA", "textvqa": "TextVQA", "docvqa": "DocVQA", "scienceqa_img": "ScienceQA"}
SETS = {
    "cot": "greedy + reasoning", "ground_cot": "greedy + grounded crops + reasoning", "zoom_cot": "greedy + zoomed crops + reasoning",
    "everything": "greedy + every 2B source", "cot_multi": "greedy + reasoning + 3 sampled reasoning",
    "ground_cot_multi": "greedy + grounded crops + reasoning + 3 sampled reasoning (2B only)",
    "esc8b": "greedy + 8B answer", "ground_cot_multi_esc8b": "2B set + 8B answer",
    "ground_cot_multi_esc": "2B set + 8B answer + 2B on enlarged images", "everything_multi_esc": "every 2B source + 8B + enlarged images",
    "cot8b": "greedy + 8B reasoning", "ground_cot_multi_cot8b": "2B set + 8B reasoning",
    "ground_cot_multi_esc8b_cot8b": "2B set + 8B answer + 8B reasoning",
}
DETECTOR = {"token": "answer probability", "pre_head": "pre-generation head", "after_head": "head on the greedy answer",
            "judge": "the model's yes/no judgement", "pre_probe": "pre-generation probe", "after_probe": "probe on the greedy answer", "random": "random"}
FIX = {"4b": "Qwen3-VL-4B answer", "8b": "Qwen3-VL-8B answer", "up1.5": "2B on the image enlarged 1.5x", "up2": "2B on the image enlarged 2x"}
BASELINE = {"greedy": "2B greedy answer", "self_consistency_5": "self-consistency, 5 answers", "self_consistency_10": "self-consistency, 10 answers",
            "reasoning_vote": "vote over 4 reasoning answers", "reasoning_greedy_vote": "vote over greedy + 4 reasoning answers",
            "set_vote": "vote over the 2B candidate set", "set_judge_best": "candidate the judge likes most",
            "set_token_best": "most probable candidate", "model_4b": "4B answer on every question", "model_8b": "8B answer on every question"}


def pct(x: float) -> str:
    """A proportion as a percentage."""
    return f"{100 * x:.2f}%"


def pts(x: float) -> str:
    """A difference in signed percentage points."""
    return f"{100 * x:+.2f}"


def ci(row: dict) -> str:
    """Difference with its bootstrap interval."""
    return f"{pts(row['difference'])} ({pts(row['ci_low'])} to {pts(row['ci_high'])})"


def pval(p: float) -> str:
    """A p-value, very small values as a bound."""
    return "&lt; 0.0001" if p < 1e-4 else f"{p:.4f}"


def label(key: str) -> str:
    """Readable name of a configuration key such as T1_k0_ground_cot_multi_esc8b."""
    k, name = key.split("_k", 1)[1].split("_", 1)
    return SETS.get(name, name) + (f" + {k} samples" if k != "0" else "")


def load(name: str) -> dict:
    """One published result file."""
    return json.loads((DATA / name).read_text())


def heads(gating: dict) -> dict:
    """Gating headlines by (configuration, name)."""
    return {(h["configuration"], h["name"]): h for h in gating["headlines"]}


def per_dataset(rows: dict, oracle: dict | None = None) -> list[list[str]]:
    """Pooled and per-dataset rows of one selector's test results."""
    out = []
    for d in ("pooled", *DATASETS):
        r = rows[d]
        greedy = r["greedy"] if "greedy" in r else r["accuracy"] - r["difference"]
        out.append([DATASETS.get(d, "All four"), pct(greedy), pct(r["accuracy"]), ci(r), f"+{r['fixed']} / −{r['broken']}", pval(r["p_value"])]
                   + ([pct(oracle[d]["accuracy"])] if oracle else []))
    return out


def cost_chart(series: list[tuple[str, list[tuple[float, float]], str]], greedy: float, xlabel: str) -> str:
    """Test accuracy against cost, one line per series of (cost, accuracy) points."""
    width, height, left, top, right, bottom = 700, 320, 58, 14, 20, 46
    xmax = max(x for _, p, _ in series for x, _ in p) * 1.03
    ymin, ymax = greedy - 0.005, max(y for _, p, _ in series for _, y in p) + 0.004
    sx = lambda x: left + x / xmax * (width - left - right)
    sy = lambda y: top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Accuracy against cost">']
    tick = round(ymin, 2)
    while tick <= ymax:
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}"/>'
                     f'<text class="tick" x="{left - 6}" y="{sy(tick) + 4:.1f}" text-anchor="end">{100 * tick:.0f}%</text>')
        tick = round(tick + 0.01, 2)
    step = 2 if xmax <= 16 else 5
    for x in range(0, int(xmax) + 1, step):
        parts.append(f'<text class="tick" x="{sx(x):.1f}" y="{height - bottom + 16}" text-anchor="middle">{x}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}"/>'
                 f'<text class="axlabel" x="{(left + width - right) / 2}" y="{height - 8}" text-anchor="middle">{xlabel}</text>'
                 f'<line class="chance" x1="{left}" x2="{width - right}" y1="{sy(greedy):.1f}" y2="{sy(greedy):.1f}"/>')
    for name, points, css in series:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
        parts.append(f'<path class="ln {css}" fill="none" d="{path}"/>')
        for x, y in points[::2]:
            parts.append(f'<circle class="dot {css}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3"><title>{name}: {pct(y)} at {x:.1f}</title></circle>')
    parts.append("</svg>")
    legend = '<div class="legend">' + "".join(
        f'<span class="lg"><svg viewBox="0 0 22 10"><line class="ln {css}" x1="1" x2="21" y1="5" y2="5"/></svg>{name}</span>' for name, _, css in series) + \
        '<span class="lg"><svg viewBox="0 0 22 10"><line class="chance" x1="1" x2="21" y1="5" y2="5"/></svg>greedy</span></div>'
    return legend + "".join(parts)


def build() -> str:
    """Assemble report 13."""
    esc, cascade, base, lat = load("escalation.json"), load("cascade.json"), load("baselines.json"), load("latency.json")
    multi, nojudge = load("candidate_ranker_v3_multi.json"), load("candidate_ranker_v3_multi_drop_judge.json")
    r_esc, r_cot, best = load("candidate_ranker_v3_multi_esc.json"), load("candidate_ranker_v3_multi_esc_cot8b.json"), load("candidate_ranker_v3_multi_esc_cot8b_drop_judge.json")
    r_esc_nj = load("candidate_ranker_v3_multi_esc_drop_judge.json")
    g_multi, g_nj = load("gating_v3_multi.json"), load("gating_v3_multi_drop_judge.json")
    g_esc, g_esc_nj = load("gating_v3_multi_esc.json"), load("gating_v3_multi_esc_drop_judge.json")
    g_cot, g_best = load("gating_v3_multi_esc_cot8b.json"), load("gating_v3_multi_esc_cot8b_drop_judge.json")
    H = {name: heads(g) for name, g in (("multi", g_multi), ("nj", g_nj), ("esc", g_esc), ("esc_nj", g_esc_nj), ("cot", g_cot), ("best", g_best))}

    greedy = g_best["greedy_test"]
    bc = best["choice"]
    best_key, best_sel = bc["configuration"], bc["selector"]
    best_rows = bc["test"][best_sel]
    TWO = "T1_k0_ground_cot_multi"
    ungated = H["best"][(best_key, "correct every question (no gate)")]
    gated50 = H["best"][(best_key, "best detector at a 50% budget")]
    gated30 = H["best"][(best_key, "best detector at a 30% budget")]
    two_old = H["multi"][(TWO, "best detector at a 50% budget")]
    two_new = H["nj"][(TWO, "best detector at a 50% budget")]
    e8 = esc["fixes"]["8b"]
    route_best = next(h for h in e8["headlines"] if h["name"] == "best detector, rule and budget")
    route30 = next(h for h in e8["headlines"] if "30% budget" in h["name"])
    all8 = base["test"]["model_8b"]

    # Where this fits: every part of the correction step, what it does, what it replaces, what it connects to.
    parts = table(["Part", "What it does", "New here / replaces", "Connects to"], [
        ["0. Detector and gate", "One score per question decides whether anything beyond the greedy answer runs; threshold fixed on validation",
         "Unchanged from report 12; now also decides which questions go to a larger model", "Answer probability (free), heads of reports 02–09"],
        ["1a. 2B candidates", "Zoomed and grounded crops, contrastive decoding, reasoning, 3 sampled reasoning answers", "Unchanged (reports 10–12)", "Encoded by the 2B model"],
        ["1b. Larger-model candidates", "Qwen3-VL-8B short answer and 8B step-by-step answer for the same question; 4B answer; 2B on enlarged images",
         "<b>New</b>: a second family of fixes, from outside the 2B model", "Encoded by the 2B model like every other candidate, so the ranker reads them unchanged"],
        ["2. Encoding", "Every distinct candidate: 2B hidden state, answer probability, head and probe scores",
         "<b>The judge pass is dropped</b> (Section 8): it carried no information the ranker used", "The trained heads"],
        ["3. Ranker v3", "Scores every candidate; the top one is the answer", "Retrained with the 8B candidates and without the judge input",
         "One more source-agreement feature (the larger-model family)"],
        ["Alternative to 1–3: routing", "Send flagged questions to the larger model and keep its answer (or the more confident one)",
         "<b>New</b>: the strongest simple competitor (Sections 4–5)", "Same detector and validation thresholds as the gate"],
    ], left_cols=(0, 1, 2, 3))

    # Section 4: routing to each fix.
    route_rows = []
    for fix in ("4b", "8b", "up1.5", "up2"):
        f = esc["fixes"][fix]
        for h in f["headlines"]:
            if h["name"].startswith("best detector and rule at a 30%") or h["name"] == "best detector, rule and budget":
                t = h["test"]["pooled"]
                route_rows.append([FIX[fix], h["name"].replace("best detector and rule at a ", "").replace("best detector, rule and budget", "best on validation"),
                                   DETECTOR[h["detector"]], h["rule"], f"{100 * h['test_share']:.1f}%", f"{h['cost']:.2f}", pct(t["accuracy"]), ci(t)])
        everything = f["fix_everything"]["pooled"]  # the fix's answer on every question, without the 2B model
        route_rows.append([FIX[fix], "every question", "", "replace", "100%", f"{esc['fix_cost'][fix]:.2f}", pct(everything["accuracy"]), ci(everything)])
    detq = [[DETECTOR[d], f"{v['greedy_correct']:.3f}", f"{v['fix_helps']:.3f}"] for d, v in e8["detector_quality_test"].items()]

    # Section 5: larger-model answers as ranker candidates.
    cand_rows = []
    for name, ranker, H_, keys in (("v3, sampled reasoning", multi, H["multi"], (TWO,)),
                                   ("v3 + 8B and enlarged images", r_esc, H["esc"], ("T1_k0_esc8b", "T1_k0_ground_cot_multi_esc8b", "T1_k0_ground_cot_multi_esc")),
                                   ("v3 + 8B reasoning", r_cot, H["cot"], ("T1_k0_cot8b", "T1_k0_ground_cot_multi_cot8b", "T1_k0_ground_cot_multi_esc8b_cot8b")),
                                   ("v3 + 8B reasoning, no judge", best, H["best"], ("T1_k0_esc8b", "T1_k0_cot8b", "T1_k0_ground_cot_multi_cot8b", "T1_k0_ground_cot_multi_esc8b_cot8b"))):
        for key in keys:
            if (key, "correct every question (no gate)") not in H_:
                continue
            u, g = H_[(key, "correct every question (no gate)")], H_[(key, "best detector at a 50% budget")]
            cand_rows.append([name, label(key), pct(u["test"]["pooled"]["accuracy"]), ci(u["test"]["pooled"]), f"{u['test_passes']:.1f}",
                              pct(g["test"]["pooled"]["accuracy"]), f"{g['test_passes']:.1f}", pct(ranker["test"][key]["oracle"]["pooled"]["accuracy"])])

    # Section 6: cascade.
    cas_rows = []
    for h in cascade["headlines"]:
        for route in ("direct", "cascade"):
            if route in h:
                r = h[route]
                diff = ci(h["cascade_minus_direct"]) if route == "cascade" and "cascade_minus_direct" in h else ""
                cas_rows.append([f"{h['cap']:g}", "2B → 8B" if route == "direct" else "2B → 4B → 8B", DETECTOR[r["detector"]], r["rule"],
                                 f"{100 * r['test_to4']:.1f}% / {100 * r['test_to8']:.1f}%", f"{r['test_cost']:.2f}", pct(r["test"]["pooled"]["accuracy"]),
                                 ci(r["test"]["pooled"]), diff])

    # Section 7: baselines.
    base_rows = [[BASELINE.get(n, n), f"{e['calls']:.1f}", f"{e['passes']:.1f}", pct(e["pooled"]["accuracy"]), ci(e["pooled"]),
                  f"+{e['pooled']['fixed']} / −{e['pooled']['broken']}"] + [pts(e[d]["difference"]) for d in DATASETS]
                 for n, e in sorted(base["test"].items(), key=lambda kv: (kv[1]["calls"], kv[0]))]

    # Section 8: ranker inputs.
    abl_rows = []
    full_row = multi["test"][TWO]["ranker_v3_scalars_ds"]["pooled"]
    full = full_row["accuracy"]
    abl_rows.append(["none (all inputs)", pct(full), "", ci(full_row)])
    for group, text in (("agreement", "source agreement"), ("token", "answer probability"), ("head", "head and probe scores"), ("judge", "the judge")):
        r = load(f"candidate_ranker_v3_multi_drop_{group}.json")["test"][TWO]["ranker_v3_scalars_ds"]["pooled"]
        abl_rows.append([text, pct(r["accuracy"]), pts(r["accuracy"] - full), ci(r)])
    nj_rows = []
    for name, (with_judge, without) in (("2B only (" + label(TWO) + ")", (H["multi"], H["nj"])),
                                        ("2B set + 8B answer + 8B reasoning", (H["cot"], H["best"]))):
        key = TWO if name.startswith("2B only") else "T1_k0_ground_cot_multi_esc8b_cot8b"
        for setting in ("correct every question (no gate)", "best detector at a 50% budget"):
            a, b = with_judge[(key, setting)], without[(key, setting)]
            nj_rows.append([name, setting.replace("best detector at a 50% budget", "gated at 50%"), pct(a["test"]["pooled"]["accuracy"]), f"{a['test_passes']:.2f}",
                            pct(b["test"]["pooled"]["accuracy"]), f"{b['test_passes']:.2f}"])

    # Section 9: leave-one-dataset-out.
    lodo_rows = []
    for d in DATASETS:
        held = load(f"candidate_ranker_v3_multi_holdout_{d}.json")["test"][TWO]["ranker_v3_scalars_nods"][d]
        inside = multi["test"][TWO]["ranker_v3_scalars_nods"][d] if "ranker_v3_scalars_nods" in multi["test"][TWO] else multi["test"][TWO]["ranker_v3_scalars_ds"][d]
        lodo_rows.append([DATASETS[d], pct(held["greedy"]), ci(inside), ci(held), f"{100 * held['difference'] / inside['difference']:.0f}%" if inside["difference"] > 0 else ""])

    # Section 10: latency.
    comp = lat["components"]
    lat_names = {"greedy": "2B short answer", "samples_10": "10 sampled 2B answers (one batched call)", "reasoning": "2B step-by-step answer",
                 "reasoning_x3": "3 sampled 2B step-by-step answers (one call)", "grounding": "grounded crops (box + 2 answers)", "zoom": "one zoomed-crop answer",
                 "vcd": "one contrastive-decoding answer", "encode": "encoding one candidate", "judge": "judging one candidate",
                 "greedy_8b": "8B short answer", "reasoning_8b": "8B step-by-step answer"}
    lat_rows = [[lat_names.get(k, k), f"{v['mean']:.3f}", f"{v['median']:.3f}", f"{v['p90']:.3f}", f"{v['mean'] / comp['greedy']['mean']:.1f}x"] for k, v in comp.items()]

    repro = [[f"{tag}", label(k), pct(r["reported"]), pct(r["ours"]), str(r["questions_different_test"]), "yes" if r["match"] else "no"]
             for tag, g in (("8B, enlarged images", g_esc), ("+ 8B reasoning", g_cot), ("+ 8B reasoning, no judge", g_best))
             for k, r in g["reproduction"].items() if "esc" in k or "cot8b" in k]

    curves = g_best["curves"]["test"]
    chart = cost_chart([
        ("ranker: 2B set + 8B answer + 8B reasoning (no judge), answer-probability gate", [(r["passes"], r["accuracy"]) for r in curves["T1_k0_ground_cot_multi_esc8b_cot8b"]["token"]], "c-uj"),
        ("ranker: greedy + 8B answer (no judge)", [(r["passes"], r["accuracy"]) for r in curves["T1_k0_esc8b"]["token"]], "c-up"),
        ("ranker: 2B candidates only (no judge)", [(r["passes"], r["accuracy"]) for r in g_nj["curves"]["test"][TWO]["token"]], "c-sv"),
        ("routing to 8B, keep the more confident answer", [(1 + 4 * b, a) for b, a in zip(esc["budgets"], e8["curves"]["test"]["token"]["confident"])] if "curves" in e8 and "test" in e8["curves"] else [], "c-tok"),
    ], greedy, "model passes per question (greedy = 1; an 8B answer counts 4)")

    toc = [("summary", "Summary"), ("pipeline", "Where this fits"), ("result", "Main result"), ("routing", "Routing to a larger model"),
           ("candidates", "Larger-model answers as candidates"), ("cascade", "Cascade"), ("baselines", "Baselines"), ("inputs", "Which inputs matter"),
           ("holdout", "Unseen training dataset"), ("latency", "Measured time"), ("checks", "Checks"), ("limits", "Limits"), ("progress", "In progress"),
           ("reproduce", "Reproduce")]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Escalation Report</title><style>{CSS}</style></head><body><main>
<h1>Approach 1, report 13: larger models as fixes, and as candidates for the ranker</h1>
<p class="sub">The 2B model's own corrections (reports 10–12) reach {pct(two_old['test']['pooled']['accuracy'])}. This report adds fixes from outside the
2B model: the answers of Qwen3-VL-8B and 4B, and the 2B model on enlarged images. They are used two ways: on their own (route the
least-confident questions to the larger model) and as extra candidates for the ranker. Qwen3-VL-2B stays the detector and the encoder of every
candidate. Full test splits, {best['questions']['test']:,} questions; every choice made on validation ({best['questions']['val']:,} questions) and
test read once; rankers trained on {best['questions']['train']:,} training questions.</p>
<p class="meta">Builds on report 12 (ranker v3, gating). Branch <code>feature/gating-escalation</code>, merged into <code>main</code>.</p>
<div class="toc">{''.join(f'<a href="#{a}">{t}</a>' for a, t in toc)}</div>

<h2 id="summary">1. Summary</h2>
<div class="stats">
<div class="stat"><b>{pct(greedy)} → {pct(ungated['test']['pooled']['accuracy'])}</b><span>test accuracy, greedy to the ranker with the 8B answers as candidates</span></div>
<div class="stat"><b>{pts(ungated['test']['pooled']['difference'])} pts</b><span>95% interval {pts(ungated['test']['pooled']['ci_low'])} to {pts(ungated['test']['pooled']['ci_high'])}</span></div>
<div class="stat"><b>{pct(gated50['test']['pooled']['accuracy'])} at {gated50['test_passes']:.1f} passes</b><span>gated: corrects {100 * gated50['test_share']:.0f}% of questions (ungated: {ungated['test_passes']:.1f})</span></div>
<div class="stat"><b>{pct(route_best['test']['pooled']['accuracy'])}</b><span>routing alone to 8B, no ranker ({route_best['cost']:.1f} passes)</span></div>
</div>
<div class="card summary"><ul>
<li><b>The 8B short answer and the 8B step-by-step answer, added as candidates, lift the ranker to {ci(ungated['test']['pooled'])} points</b>
({pct(ungated['test']['pooled']['accuracy'])}), against {ci(two_new['test']['pooled'])} with 2B candidates only. Every dataset gains: ChartQA
{pts(best_rows['chartqa_full']['difference'])}, TextVQA {pts(best_rows['textvqa']['difference'])}, DocVQA {pts(best_rows['docvqa']['difference'])},
ScienceQA {pts(best_rows['scienceqa_img']['difference'])}. Gated at 50% of questions it keeps {pct(gated50['test']['pooled']['accuracy'])} at
{gated50['test_passes']:.1f} passes per question.</li>
<li><b>Routing alone is a strong competitor.</b> Sending the least-confident questions to the 8B model and keeping the more confident of the two
answers gives {ci(route_best['test']['pooled'])} points ({route_best['cost']:.1f} passes); 30% routed gives {ci(route30['test']['pooled'])} at
{route30['cost']:.1f} passes. With the 8B short answer as its only extra candidate the ranker matches routing; it pulls ahead only when it also has the
8B reasoning answer and the 2B candidates (about {100 * (ungated['test']['pooled']['accuracy'] - route_best['test']['pooled']['accuracy']):.1f} points over the best routing).</li>
<li><b>A 2B → 4B → 8B cascade helps only at the smallest budget.</b> Under 1.5 passes per question it beats routing straight to 8B by
{ci(cascade['headlines'][0]['cascade_minus_direct'])} points (8B does not fit that budget); from 2 passes up the two are equal within noise (Section 6).</li>
<li><b>The judge pass can be dropped.</b> Removing the model's yes/no judgement from the ranker's inputs changes accuracy by
{abl_rows[4][2]} points and removes one of the two scoring passes per candidate: the 2B-only ranker keeps {pct(two_new['test']['pooled']['accuracy'])}
at {two_new['test_passes']:.1f} passes instead of {two_old['test_passes']:.1f}. The input that matters is source agreement: without it the ranker loses
{abl_rows[1][2].lstrip('+-')} points (Section 8).</li>
<li><b>The ranker transfers to a dataset it never trained on.</b> Trained on three datasets and tested on the fourth, it keeps most of its gain on
each (Section 9).</li>
<li><b>Measured time changes the cost picture.</b> On one H100 an 8B short answer takes {comp['greedy_8b']['mean']:.2f} s against
{comp['greedy']['mean']:.2f} s for the 2B one, while a step-by-step answer takes {comp['reasoning']['mean']:.1f}–{comp['reasoning_8b']['mean']:.1f} s.
Counted in passes, the 8B answer is expensive and reasoning cheap; in seconds it is the other way round (Section 10).</li>
</ul></div>

<h2 id="pipeline">2. Where this fits</h2>
<p>The correction step of report 12 had a gate, 2B candidates, an encoding pass with a judge, and the ranker. This report adds one family of
candidates from outside the 2B model, removes the judge, and tests the simplest alternative to the whole step: give the question to a larger model.</p>
{parts}

<h2 id="result">3. Main result</h2>
<p>Chosen on validation: <b>{label(best_key)}</b>, selector <code>{best_sel}</code>, ranker trained without the judge input.</p>
{table(["Dataset", "Greedy", "Ranker", "Gain (95% interval)", "Fixed / broken", "p (McNemar)", "Best possible pick"], per_dataset(best_rows, bc['test']['oracle']), best_rows=(0,))}
{figure("fig-cost", "Figure 1. Test accuracy against cost as the gate opens", chart,
        "Each ranker line runs from 0% of questions corrected (greedy, 1 pass) to 100% (no gate). The routing line sends the same shares of questions to the 8B model. Passes count an 8B answer as 4; Section 10 gives the measured seconds.")}

<h2 id="routing">4. Routing to a larger model (no ranker)</h2>
<p>The detector flags the least-confident questions (threshold from validation, as for the gate); a flagged question gets the fix's answer
(rule <i>replace</i>) or keeps whichever of the two answers has the higher answer probability (rule <i>confident</i>). Detector, rule and budget are
chosen on validation. Cost in 2B passes: a 4B answer counts 2, an 8B answer 4, an enlarged image the pixel ratio.</p>
{table(["Fix", "Setting", "Detector", "Rule", "Routed (test)", "Passes", "Test accuracy", "Gain (95% interval)"], route_rows, left_cols=(0, 1, 2, 3))}
<p>Which detector finds the questions a fix will help? Test AUROC for "the 2B answer is right" and for "the 8B answer is right where the 2B one is wrong":</p>
{table(["Detector", "2B answer right", "8B fix helps"], detq, left_cols=(0,))}
<p>The trained head is the better judge of whether the 2B answer is right; for finding the questions the 8B model fixes, the answer probability is as good.
Enlarging the image helps little (about a point at best): the processor already reads images at their native size.</p>

<h2 id="candidates">5. Larger-model answers as ranker candidates</h2>
<p>The 8B answers are encoded by the 2B model exactly like the 2B candidates, and the ranker gets one more source-agreement feature (the share of
larger-model proposals equal to the candidate). The rankers are retrained on the 12,000 training questions, for which the 8B answers were generated too.</p>
{table(["Ranker", "Candidates", "No gate", "Gain (95% interval)", "Passes", "Gated 50%", "Passes", "Best possible pick"], cand_rows, left_cols=(0, 1))}
<p>The 8B short answer alone already carries most of the gain; the 8B reasoning answer adds to it when the 2B candidates are present, and the
best possible pick from that set is close to 95%. The enlarged-image answers do not help the ranker.</p>

<h2 id="cascade">6. A 2B → 4B → 8B cascade</h2>
<p>Step 1: the 2B detector flags a share of questions, which get a 4B answer. Step 2: of those, the ones whose 4B answer is least probable also get
an 8B answer. Every setting (detector, rule, both shares) is chosen on validation under each average cost and compared with routing straight to 8B
under the same cost, on the same questions.</p>
{table(["Cost cap", "Route", "Detector", "Rule", "To 4B / to 8B (test)", "Passes", "Test accuracy", "Gain (95% interval)", "Cascade minus direct"], cas_rows, left_cols=(1, 2, 3))}
<p>The 4B step pays only when there is no budget for 8B. Once 8B fits, sending its share of questions straight to 8B is as good, so the cascade
adds a model without adding accuracy.</p>

<h2 id="baselines">7. Training-free baselines on the same questions</h2>
<p>Every baseline uses the same questions and, where it uses candidates, the same candidates as the ranker. None has a parameter chosen on test.</p>
{table(["Method", "Calls", "Passes", "Test accuracy", "Gain (95% interval)", "Fixed / broken"] + list(DATASETS.values()), base_rows, left_cols=(0,))}
<p>Sampling more 2B answers and voting barely moves accuracy; voting over the reasoning answers helps ChartQA and ScienceQA and hurts TextVQA and
DocVQA; picking the candidate the model's own judge likes most is clearly worse than the greedy answer. The learned ranker over the same 2B
candidates ({pct(two_new['test']['pooled']['accuracy'])}) is well above every vote. Running the 8B model on every question
({pct(all8['pooled']['accuracy'])}, 4 passes) is the strongest baseline; routing only the flagged questions to it reaches more at less cost.</p>

<h2 id="inputs">8. Which ranker inputs matter</h2>
<p>The 2B-only ranker (sampled reasoning) retrained with one group of inputs set to zero in training and evaluation, everything else equal;
{label(TWO)}, test:</p>
{table(["Inputs removed", "Test accuracy", "Change", "Gain over greedy (95% interval)"], abl_rows, left_cols=(0,))}
<p>Source agreement is what the ranker relies on; the answer probability and the head add a little; the judge adds nothing. Since the judge costs
one forward pass per candidate, the rankers are retrained without it and gated with one scoring pass per candidate:</p>
{table(["Candidates", "Setting", "With judge", "Passes", "Without judge", "Passes"], nj_rows, left_cols=(0, 1))}

<h2 id="holdout">9. A dataset the ranker never trained on</h2>
<p>The 2B-only ranker (no dataset input) retrained four times, each time without one dataset in training and in the learning-rate choice, then
tested on that dataset ({label(TWO)}). For comparison, the same ranker trained on all four.</p>
{table(["Held-out dataset", "Greedy", "Gain, trained on all four", "Gain, never trained on it", "Share kept"], lodo_rows)}

<h2 id="latency">10. Measured time</h2>
<p>Mean, median and 90th-percentile seconds per question of every component on one {lat['gpu']}, on the first {lat['per_dataset']} test questions of
each dataset (batch size 1; the first question of each run is a warm-up and not timed).</p>
{table(["Component", "Mean (s)", "Median (s)", "90th percentile (s)", "Against a 2B short answer"], lat_rows, left_cols=(0,))}
<p>Two of the pass-counting conventions do not match measured time. An 8B short answer counts as 4 passes but takes
{comp['greedy_8b']['mean'] / comp['greedy']['mean']:.1f} times a 2B answer: at batch size 1 the GPU is far from saturated, so time grows much
more slowly than parameter count. A step-by-step answer counts as 1 call but takes {comp['reasoning']['mean'] / comp['greedy']['mean']:.0f} times a
short answer, because it generates up to 256–384 tokens. In seconds, an 8B answer costs about as much as one zoomed-crop answer and is the most
effective single fix, while the reasoning candidates are the most expensive; the gating study with seconds as the cost is in progress (Section 13).</p>

<h2 id="checks">11. Checks</h2>
<ul><li><b>Gating uses the exact saved rankers.</b> Before gating, each configuration's ungated accuracy is recomputed and compared with the training
run's report on validation and test; the script stops otherwise:</li></ul>
{table(["Ranker", "Candidates", "Reported (test)", "Gating script, no gate", "Questions different", "Match"], repro, left_cols=(0, 1))}
<ul>
<li><b>Complete inputs.</b> Every validation and test question has its 4B and 8B answers (0 missing), and the escalation candidates cover all
12,000 training questions.</li>
<li><b>Tests.</b> Unit tests for the routing simulation, the cascade, the escalation candidates, the gate and the rankers pass on a GPU node before
the analyses run.</li>
</ul>

<h2 id="limits">12. Limits</h2>
<ul>
<li>The larger models are of the same family as the 2B model; whether a different family helps as much is not tested.</li>
<li>Pass counts are a convention; Section 10 shows they misstate the relative cost of 8B answers and reasoning.</li>
<li>Latency is measured at batch size 1 on one GPU type; batched serving would change the ratios.</li>
<li>Four datasets, all in-domain for the ranker except in Section 9.</li>
</ul>

<h2 id="progress">13. In progress</h2>
<ul>
<li>Qwen3-VL-4B as the base model: the whole pipeline (pools, both correction rounds, sampled reasoning, ranker, gating) with 4B in place of 2B.</li>
<li>Transfer to datasets the pipeline never saw (AI2D, InfographicVQA), with the frozen rankers and in-domain thresholds, including the best ranker
of this report.</li>
<li>For the best ranker: two further seed sets, leave-one-dataset-out, and training on 250, 750 and 1,500 questions per dataset.</li>
<li>Gating with measured seconds as the cost axis.</li>
</ul>

<h2 id="reproduce">14. Reproduce</h2>
<pre>bash slurm/run_escalation_pipeline.sh            # 8B, 4B and enlarged-image answers on val/test; routing study
bash slurm/run_escalation_ranker_pipeline.sh     # the same on the train subset, as candidates; ranker; gating
bash slurm/run_reasoning_8b_pipeline.sh          # 8B reasoning as a candidate; ranker; gating
bash slurm/run_next_steps_pipeline.sh            # cascade; rankers without the judge; gating with one scoring pass
sbatch slurm/compute_baselines.sbatch            # training-free baselines
sbatch slurm/measure_latency.sbatch              # seconds per component on an H100
python -m experiments.approach1_latent_self_correction.build_report_13   # this report</pre>
<p class="small">Data: <code>results/approach1/04_sweep_data/</code> (escalation, cascade, baselines, latency, candidate_ranker_v3_multi*, gating_v3_multi*).
Code: <code>escalation.py</code>, <code>escalation_candidates.py</code>, <code>analyze_escalation.py</code>, <code>analyze_cascade.py</code>,
<code>compute_baselines.py</code>, <code>measure_latency.py</code>, <code>train_ranker_v3.py</code>, <code>gate_ranker_v3.py</code>.</p>
</main></body></html>"""


def main() -> None:
    """Write the report."""
    parser = argparse.ArgumentParser(description="Build report 13.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(build(), encoding="utf-8")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
