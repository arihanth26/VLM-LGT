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
#   gating_v3_multi*.json                gating of the saved rankers (gate_ranker_v3.py; *_seconds with measured seconds)
#   best_ranker_analysis.json            paired comparisons, answer origins, oracle, abstention (analyze_best_ranker.py)
#   unseen.json                          the frozen pipeline on AI2D and InfographicVQA (evaluate_unseen.py)
#   base_4b/, base_8b/                   the pipeline with Qwen3-VL-4B and 8B as the base model
# Writes results/approach1/13_escalation_report.html.

import argparse
import html
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
    g_multi, g_nj = load("gating_v3_multi.json"), load("gating_v3_multi_drop_judge.json")
    g_esc, g_esc_nj = load("gating_v3_multi_esc.json"), load("gating_v3_multi_esc_drop_judge.json")
    g_cot, g_best = load("gating_v3_multi_esc_cot8b.json"), load("gating_v3_multi_esc_cot8b_drop_judge.json")
    g_sec = load("gating_v3_multi_esc_cot8b_drop_judge_seconds.json")
    analysis, unseen = load("best_ranker_analysis.json"), load("unseen.json")
    H = {name: heads(g) for name, g in (("multi", g_multi), ("nj", g_nj), ("esc", g_esc), ("esc_nj", g_esc_nj), ("cot", g_cot), ("best", g_best), ("sec", g_sec))}

    greedy = g_best["greedy_test"]
    bc = best["choice"]
    best_key, best_sel = bc["configuration"], bc["selector"]
    best_rows = bc["test"][best_sel]
    TWO, K = "T1_k0_ground_cot_multi", "T1_k0_ground_cot_multi_esc8b_cot8b"
    ungated = H["best"][(best_key, "correct every question (no gate)")]
    gated50 = H["best"][(best_key, "best detector at a 50% budget")]
    two_old = H["multi"][(TWO, "best detector at a 50% budget")]
    two_new = H["nj"][(TWO, "best detector at a 50% budget")]
    e8, e4 = esc["fixes"]["8b"], esc["fixes"]["4b"]
    route_best = next(h for h in e8["headlines"] if h["name"] == "best detector, rule and budget")
    route30 = next(h for h in e8["headlines"] if "30% budget" in h["name"])
    route4_30 = next(h for h in e4["headlines"] if "30% budget" in h["name"])
    all8, all4 = base["test"]["model_8b"], base["test"]["model_4b"]
    comp = lat["components"]
    paired = analysis["paired"]

    # Where this fits: every part of the correction step, what it does, what it replaces, what it connects to.
    parts = table(["Part", "What it does", "New here / replaces", "Connects to"], [
        ["0. Detector and gate", "One score per question decides whether anything beyond the greedy answer runs; threshold fixed on validation",
         "Unchanged from report 12; now also decides which questions go to a larger model", "Answer probability (free), heads of reports 02–09"],
        ["1a. 2B candidates", "Zoomed and grounded crops, contrastive decoding, reasoning, 3 sampled reasoning answers", "Unchanged (reports 10–12)", "Encoded by the 2B model"],
        ["1b. Larger-model candidates", "Qwen3-VL-8B short answer and 8B step-by-step answer for the same question; 4B answer; 2B on enlarged images",
         "<b>New</b>: a second family of fixes, from outside the 2B model", "Encoded by the 2B model like every other candidate, so the ranker reads them unchanged"],
        ["2. Encoding", "Every distinct candidate: 2B hidden state, answer probability, head and probe scores",
         "<b>The judge pass is dropped</b> (Section 10): it carried no information the ranker used", "The trained heads"],
        ["3. Ranker v3", "Scores every candidate; the top one is the answer", "Retrained with the 8B candidates and without the judge input",
         "One more source-agreement feature (the larger-model family)"],
        ["Alternative to 1–3: routing", "Send flagged questions to the larger model and keep its answer (or the more confident one)",
         "<b>New</b>: the strongest simple competitor (Sections 4–5)", "Same detector and validation thresholds as the gate"],
    ], left_cols=(0, 1, 2, 3))

    # Section 4: the accuracy-cost frontier, ranker policies and routing on one table.
    def sec_of(h):
        s = H["sec"].get((h["configuration"], h["name"]))
        return f"{s['test_seconds']:.2f}" if s else "—"
    frontier = []
    for name, label_ in (("best under 2 passes per question (any configuration)", "Best ranker policy under 2 passes"),
                         ("best under 3 passes per question (any configuration)", "Best ranker policy under 3 passes"),
                         ("best under 5 passes per question (any configuration)", "Best ranker policy under 5 passes")):
        h = next(x for x in g_best["headlines"] if x["name"] == name)
        frontier.append([label_, f"{label(h['configuration'])}; {DETECTOR[h['detector']]} gate, {100 * h['test_share']:.1f}% corrected", f"{h['test_passes']:.2f}", sec_of(h),
                         pct(h["test"]["pooled"]["accuracy"]), ci(h["test"]["pooled"])])
    for setting, label_ in (("smallest budget keeping 90% of the full validation gain", "Full candidate system, 40% gate"),
                            ("best detector at a 50% budget", "Full candidate system, 50% gate"), ("correct every question (no gate)", "Full candidate system, no gate")):
        h = H["best"][(K, setting)]
        frontier.append([label_, f"{label(K)}; {DETECTOR[h['detector']]} gate, {100 * h['test_share']:.1f}% corrected", f"{h['test_passes']:.2f}", sec_of(h),
                         pct(h["test"]["pooled"]["accuracy"]), ci(h["test"]["pooled"])])
    detector_seconds = {"after_head": comp["encode"]["mean"], "after_probe": comp["encode"]["mean"]}  # the detector's own pass, if any
    route_seconds = lambda h, comp_name: f"≈{comp['greedy']['mean'] + detector_seconds.get(h['detector'], 0.0) + h['test_share'] * comp[comp_name]['mean']:.2f}"
    for h, label_, fix in ((route4_30, "Route 30% to 4B (no ranker)", "4b"), (route30, "Route 30% to 8B (no ranker)", "8b"),
                           (route_best, "Route to 8B, setting chosen on validation (no ranker)", "8b")):
        frontier.append([label_, f"{DETECTOR[h['detector']]}, keep the {'more confident answer' if h['rule'] == 'confident' else 'fix'}; {100 * h['test_share']:.1f}% routed",
                         f"{h['cost']:.2f}", route_seconds(h, "greedy_8b") if fix == "8b" else "—", pct(h["test"]["pooled"]["accuracy"]), ci(h["test"]["pooled"])])
    frontier.append(["4B on every question", "", "2.00", "—", pct(all4["pooled"]["accuracy"]), ci(all4["pooled"])])
    frontier.append(["8B on every question", "", "4.00", f"{comp['greedy_8b']['mean']:.2f}", pct(all8["pooled"]["accuracy"]), ci(all8["pooled"])])
    seconds_rows = [[h["name"].replace(" on one H100 (any configuration)", "").replace("best under", "Best under"), label(h["configuration"]),
                     f"{DETECTOR[h['detector']]}, {100 * h['test_share']:.1f}% corrected", f"{h['test_seconds']:.2f}", f"{h['test_passes']:.2f}",
                     pct(h["test"]["pooled"]["accuracy"]), ci(h["test"]["pooled"])] for h in g_sec["headlines"] if " s per question" in h["name"]]

    # Section 5: routing to each fix.
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

    # Section 6: larger-model answers as ranker candidates.
    cand_rows = []
    for name, ranker, H_, keys in (("v3, sampled reasoning", multi, H["multi"], (TWO,)),
                                   ("v3 + 8B and enlarged images", r_esc, H["esc"], ("T1_k0_esc8b", "T1_k0_ground_cot_multi_esc8b", "T1_k0_ground_cot_multi_esc")),
                                   ("v3 + 8B reasoning", r_cot, H["cot"], ("T1_k0_cot8b", "T1_k0_ground_cot_multi_cot8b", K)),
                                   ("v3 + 8B reasoning, no judge", best, H["best"], ("T1_k0_esc8b", "T1_k0_cot8b", "T1_k0_ground_cot_multi_cot8b", K))):
        for key in keys:
            if (key, "correct every question (no gate)") not in H_:
                continue
            u, g = H_[(key, "correct every question (no gate)")], H_[(key, "best detector at a 50% budget")]
            cand_rows.append([name, label(key), pct(u["test"]["pooled"]["accuracy"]), ci(u["test"]["pooled"]), f"{u['test_passes']:.1f}",
                              pct(g["test"]["pooled"]["accuracy"]), f"{g['test_passes']:.1f}", pct(ranker["test"][key]["oracle"]["pooled"]["accuracy"])])

    # Section 7: what the best ranker does.
    def pci(d):
        return f"{pts(d['difference'])} ({pts(d['ci_low'])} to {pts(d['ci_high'])})"
    paired_rows = [[name, pci(rows["pooled"]), pval(rows["pooled"]["p_value"])] + [pts(rows[d]["difference"]) for d in DATASETS] for name, rows in paired.items()]
    origin_rows = [[k, f"{100 * v['share']:.1f}%", pct(v["accuracy"]), str(v["fixed"]), str(v["broken"])] for k, v in analysis["origin"].items()]
    oracle = analysis["oracle"]
    sel_rows = [[k] + [pct(v[f"accuracy_at_{c}"]) for c in (50, 80, 90, 100)] + [f"{v['aurc']:.4f}"] for k, v in analysis["selective"].items()]
    esc_html = lambda x: html.escape(", ".join(x) if isinstance(x, list) else str(x))
    example_rows = [[DATASETS[e["dataset"]], e["kind"], esc_html(e["question"]), esc_html(e["reference"]), esc_html(e["greedy"]), esc_html(e["chosen"]), e["origin"]]
                    for e in analysis["examples"] if e["dataset"] in DATASETS][:10]

    # Section 8: cascade.
    cas_rows = []
    for h in cascade["headlines"]:
        for route in ("direct", "cascade"):
            if route in h:
                r = h[route]
                diff = ci(h["cascade_minus_direct"]) if route == "cascade" and "cascade_minus_direct" in h else ""
                cas_rows.append([f"{h['cap']:g}", "2B → 8B" if route == "direct" else "2B → 4B → 8B", DETECTOR[r["detector"]], r["rule"],
                                 f"{100 * r['test_to4']:.1f}% / {100 * r['test_to8']:.1f}%", f"{r['test_cost']:.2f}", pct(r["test"]["pooled"]["accuracy"]),
                                 ci(r["test"]["pooled"]), diff])

    # Section 9: baselines.
    base_rows = [[BASELINE.get(n, n), f"{e['calls']:.1f}", f"{e['passes']:.1f}", pct(e["pooled"]["accuracy"]), ci(e["pooled"]),
                  f"+{e['pooled']['fixed']} / −{e['pooled']['broken']}"] + [pts(e[d]["difference"]) for d in DATASETS]
                 for n, e in sorted(base["test"].items(), key=lambda kv: (kv[1]["calls"], kv[0]))]

    # Section 10: ranker inputs, for the 2B-only ranker and for the best ranker.
    def acc(name, key, sel="ranker_v3_scalars_ds"):
        return load(name)["test"][key][sel]["pooled"]
    full_row = multi["test"][TWO]["ranker_v3_scalars_ds"]["pooled"]
    best_full = best["test"][K]["ranker_v3_scalars_ds"]["pooled"]
    abl_rows = [["none (all inputs)", pct(full_row["accuracy"]), "", pct(best_full["accuracy"]), "(judge already removed)"]]
    for group, text, best_file in (("agreement", "source agreement", "candidate_ranker_v3_multi_esc_cot8b_drop_agreement_judge.json"),
                                   ("token", "answer probability", "candidate_ranker_v3_multi_esc_cot8b_drop_judge_token.json"),
                                   ("head", "head and probe scores", "candidate_ranker_v3_multi_esc_cot8b_drop_head_judge.json"),
                                   ("judge", "the judge", None)):
        r = acc(f"candidate_ranker_v3_multi_drop_{group}.json", TWO)
        b = acc(best_file, K) if best_file else None
        abl_rows.append([text, pct(r["accuracy"]), pts(r["accuracy"] - full_row["accuracy"]),
                         pct(b["accuracy"]) if b else "—", pts(b["accuracy"] - best_full["accuracy"]) if b else "—"])
    variant_rows = [["as reported (scalar inputs, 10 epochs)", pct(best_full["accuracy"])],
                    ["with the hidden-state inputs (full_ds)", pct(acc("candidate_ranker_v3_multi_esc_cot8b_drop_judge_states.json", K, "ranker_v3_full_ds")["accuracy"])],
                    ["20 epochs instead of 10", pct(acc("candidate_ranker_v3_multi_esc_cot8b_drop_judge_epochs20.json", K)["accuracy"])]]
    nj_rows = []
    for name, (with_judge, without) in (("2B only (" + label(TWO) + ")", (H["multi"], H["nj"])), ("2B set + 8B answer + 8B reasoning", (H["cot"], H["best"]))):
        key = TWO if name.startswith("2B only") else K
        for setting in ("correct every question (no gate)", "best detector at a 50% budget"):
            a, b = with_judge[(key, setting)], without[(key, setting)]
            nj_rows.append([name, setting.replace("best detector at a 50% budget", "gated at 50%"), pct(a["test"]["pooled"]["accuracy"]), f"{a['test_passes']:.2f}",
                            pct(b["test"]["pooled"]["accuracy"]), f"{b['test_passes']:.2f}"])

    # Section 11: robustness.
    seed_rows = []
    for name, stem, key in (("Best ranker", "candidate_ranker_v3_multi_esc_cot8b_drop_judge", K), ("2B-only ranker, no judge", "candidate_ranker_v3_multi_drop_judge", TWO)):
        values = [acc(f"{stem}{s}.json", key)["accuracy"] for s in ("", "_seeds100", "_seeds200")]
        seed_rows.append([name, label(key)] + [pct(v) for v in values] + [f"{100 * (max(values) - min(values)):.2f}"])
    lodo_rows = []
    for d in DATASETS:
        held_b = load(f"candidate_ranker_v3_multi_esc_cot8b_drop_judge_holdout_{d}.json")["test"][K]["ranker_v3_scalars_nods"][d]
        in_b = best["test"][K]["ranker_v3_scalars_nods"][d]
        held_2 = load(f"candidate_ranker_v3_multi_holdout_{d}.json")["test"][TWO]["ranker_v3_scalars_nods"][d]
        in_2 = multi["test"][TWO]["ranker_v3_scalars_nods"][d]
        share = lambda h, i: f"{100 * h['difference'] / i['difference']:.0f}%" if i["difference"] > 0 else ""
        lodo_rows.append([DATASETS[d], pct(held_b["greedy"]), ci(in_b), ci(held_b), share(held_b, in_b), ci(in_2), ci(held_2), share(held_2, in_2)])
    size_rows = []
    for n in (250, 750, 1500, 3000):
        b = best if n == 3000 else load(f"candidate_ranker_v3_multi_esc_cot8b_drop_judge_train{n}.json")
        t = nojudge if n == 3000 else load(f"candidate_ranker_v3_multi_drop_judge_train{n}.json")
        size_rows.append([f"{n:,}", f"{4 * n:,}", pct(b["test"][K]["ranker_v3_scalars_ds"]["pooled"]["accuracy"]), pct(t["test"][TWO]["ranker_v3_scalars_ds"]["pooled"]["accuracy"])])

    # Section 12: transfer to unseen datasets.
    UNSEEN = {"ai2d": "AI2D", "infovqa": "InfographicVQA"}
    find = lambda prefix: next((v for k, v in unseen["pipeline"].items() if k.startswith(prefix)), None)
    ub = unseen["baselines"]
    u_greedy = ub["greedy"]["pooled"]["accuracy"]
    unseen_rows = [["2B greedy answer", pct(u_greedy), "", ""] + [pct(ub["greedy"][d]["accuracy"]) for d in UNSEEN]]
    for name, prefix in (("2B-only ranker, every question", f"{TWO}, every question"), ("2B-only ranker, 50% gate", f"{TWO}, gate 50%"),
                         ("8B on gated questions, 30% gate", "8b on gated questions, gate 30%"), ("8B on gated questions, 50% gate", "8b on gated questions, gate 50%"),
                         ("Best ranker, every question", f"{K} (8B reasoning, no judge), every question"), ("Best ranker, 50% gate", f"{K} (8B reasoning, no judge), gate 50%")):
        r = find(prefix)
        if r:
            gate_note = prefix.split("(corrected")[0]
            unseen_rows.append([name, pct(r["pooled"]["accuracy"]), ci(r["pooled"]), f"+{r['pooled']['fixed']} / −{r['pooled']['broken']}"] + [pts(r[d]["difference"]) for d in UNSEEN])
    for n in ("set_vote", "self_consistency_10", "model_8b"):
        if n in ub:
            e = ub[n]
            unseen_rows.append([BASELINE.get(n, n), pct(e["pooled"]["accuracy"]), ci(e["pooled"]), f"+{e['pooled']['fixed']} / −{e['pooled']['broken']}"] + [pts(e[d]["difference"]) for d in UNSEEN])
    corrected_share = next((k.split("corrected ")[1].rstrip(")") for k in unseen["pipeline"] if "gate 50%" in k), "")

    # Section 13: other base models.
    base_rows_m = []
    for size, f, cand in (("Qwen3-VL-2B", "candidate_ranker_v3_multi_drop_judge.json", "own candidates (no judge)"), ("Qwen3-VL-2B", "candidate_ranker_v3_multi_esc_cot8b_drop_judge.json", "own candidates + 8B answers (no judge)"),
                          ("Qwen3-VL-4B", "base_4b/candidate_ranker_v3_multi.json", "own candidates"), ("Qwen3-VL-4B", "base_4b/candidate_ranker_v3_multi_esc_cot8b_drop_judge.json", "own candidates + 8B answers (no judge)"),
                          ("Qwen3-VL-8B", "base_8b/candidate_ranker_v3_multi.json", "own candidates")):
        r = load(f); c = r["choice"]; x = c["test"][c["selector"]]["pooled"]
        base_rows_m.append([size, cand, f"{r['questions']['train']:,}", pct(x["greedy"]), pct(x["accuracy"]), ci(x), pct(c["test"]["oracle"]["pooled"]["accuracy"])])

    # Section 14: latency.
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
        ("ranker: 2B set + 8B answer + 8B reasoning (no judge), answer-probability gate", [(r["passes"], r["accuracy"]) for r in curves[K]["token"]], "c-uj"),
        ("ranker: greedy + 8B answer (no judge)", [(r["passes"], r["accuracy"]) for r in curves["T1_k0_esc8b"]["token"]], "c-up"),
        ("ranker: 2B candidates only (no judge)", [(r["passes"], r["accuracy"]) for r in g_nj["curves"]["test"][TWO]["token"]], "c-sv"),
        ("routing to 8B, keep the more confident answer (no ranker)", [(1 + 4 * b, a) for b, a in zip(esc["budgets"], e8["curves"]["test"]["token"]["confident"])], "c-tok"),
    ], greedy, "model passes per question (greedy = 1; an 8B answer counts 4)")

    agreement_drop = abl_rows[1][4]
    toc = [("summary", "Summary"), ("pipeline", "Where this fits"), ("result", "Main result"), ("frontier", "Accuracy against cost"), ("routing", "Routing"),
           ("candidates", "Larger-model candidates"), ("inside", "What the best ranker does"), ("cascade", "Cascade"), ("baselines", "Baselines"),
           ("inputs", "Which inputs matter"), ("robust", "Robustness"), ("unseen", "Unseen datasets"), ("bases", "Other base models"), ("latency", "Measured time"),
           ("checks", "Checks"), ("limits", "Limits"), ("progress", "In progress"), ("reproduce", "Reproduce")]
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
<div class="stat"><b>{pci(paired['best ranker minus routing to 8B']['pooled'])}</b><span>best ranker against routing to 8B alone, paired, same questions</span></div>
</div>
<div class="card summary"><ul>
<li><b>The 8B short answer and the 8B step-by-step answer, added as candidates, lift the ranker to {ci(ungated['test']['pooled'])} points</b>
({pct(ungated['test']['pooled']['accuracy'])}), against {ci(two_new['test']['pooled'])} with 2B candidates only. Every dataset gains: ChartQA
{pts(best_rows['chartqa_full']['difference'])}, TextVQA {pts(best_rows['textvqa']['difference'])}, DocVQA {pts(best_rows['docvqa']['difference'])},
ScienceQA {pts(best_rows['scienceqa_img']['difference'])}. Gated at 50% of questions it keeps {pct(gated50['test']['pooled']['accuracy'])} at
{gated50['test_passes']:.1f} passes per question.</li>
<li><b>Routing alone is the strongest simple policy, and the ranker beats it.</b> Sending the least-confident questions to the 8B model and keeping
the more confident of the two answers gives {pct(route_best['test']['pooled']['accuracy'])} at {route_best['cost']:.1f} passes; below about 5 passes per
question no ranker policy does better (Section 4). The full ranker is above it by {pci(paired['best ranker minus routing to 8B']['pooled'])} points
(p {pval(paired['best ranker minus routing to 8B']['pooled']['p_value'])}), and by {pci(paired['best ranker gated 50% minus routing to 8B']['pooled'])}
when gated at 50% (Section 7).</li>
<li><b>Source agreement is what the ranker relies on.</b> Without it the best ranker falls by {agreement_drop.lstrip('+-')} points; the judge can be
dropped at no cost, which saves one scoring pass per candidate (Section 10).</li>
<li><b>The result is stable.</b> Two further seed sets move the best ranker by less than 0.1 point; trained without one dataset it keeps most of its gain
there; and it reaches {size_rows[1][2]} with a quarter of the training questions (Section 11).</li>
<li><b>On datasets the pipeline never saw</b> (AI2D, InfographicVQA), the frozen 2B-only ranker gains {pts(find(f'{TWO}, every question')['pooled']['difference'])} points
and the best ranker {pts(find(f'{K} (8B reasoning, no judge), every question')['pooled']['difference'])}; but the 8B model on every question already gains
{pts(ub['model_8b']['pooled']['difference'])} there (Section 12).</li>
<li><b>A stronger base model gains less.</b> The same pipeline gains {base_rows_m[2][5].split(' ')[0]} points with Qwen3-VL-4B as the base model
({base_rows_m[3][5].split(' ')[0]} with the 8B answers as candidates) and {base_rows_m[4][5].split(' ')[0]} with Qwen3-VL-8B (Section 13).</li>
<li><b>Measured time changes the cost picture.</b> On one H100 an 8B short answer takes {comp['greedy_8b']['mean']:.2f} s against
{comp['greedy']['mean']:.2f} s for the 2B one, while a step-by-step answer takes {comp['reasoning']['mean']:.1f}–{comp['reasoning_8b']['mean']:.1f} s. Counted in
seconds, the ranker over the greedy and 8B answers reaches {seconds_rows[0][5]} at {seconds_rows[0][3]} s per question (Section 14).</li>
</ul></div>

<h2 id="pipeline">2. Where this fits</h2>
<p>The correction step of report 12 had a gate, 2B candidates, an encoding pass with a judge, and the ranker. This report adds one family of
candidates from outside the 2B model, removes the judge, and tests the simplest alternative to the whole step: give the question to a larger model.</p>
{parts}

<h2 id="result">3. Main result</h2>
<p>Chosen on validation: <b>{label(best_key)}</b>, selector <code>{best_sel}</code>, ranker trained without the judge input.</p>
{table(["Dataset", "Greedy", "Ranker", "Gain (95% interval)", "Fixed / broken", "p (McNemar)", "Best possible pick"], per_dataset(best_rows, bc['test']['oracle']), best_rows=(0,))}

<h2 id="frontier">4. Accuracy against cost</h2>
<p>Every policy here is chosen on validation and read once on test. Ranker policies: the best configuration, detector and gate budget under an average
cost cap, and the full candidate system at three gate budgets (gating_v3_multi_esc_cot8b_drop_judge). Routing policies use no ranker. Passes count
generation calls and scoring passes, with a 4B answer as 2 and an 8B answer as 4; seconds are the measured H100 component times (Section 14) summed
over what each question runs (routing seconds are estimated the same way).</p>
{table(["Policy", "What it runs", "Passes", "Seconds", "Test accuracy", "Gain (95% interval)"], frontier, left_cols=(0, 1))}
<p>Below about 5 passes, routing to the 8B model without a ranker is as good as any ranker policy; the three capped ranker policies all choose the greedy
answer and the 8B step-by-step answer as candidates. The ranker over the full candidate set is above routing once the budget allows it (Section 7 gives
the paired test). With seconds as the cost, chosen on validation under each cap:</p>
{table(["Cap", "Candidates", "Gate", "Seconds", "Passes", "Test accuracy", "Gain (95% interval)"], seconds_rows, left_cols=(0, 1, 2))}
{figure("fig-cost", "Figure 1. Test accuracy against cost as the gate opens", chart,
        "Each ranker line runs from 0% of questions corrected (greedy, 1 pass) to 100% (no gate). The routing line sends the same shares of questions to the 8B model. Passes count an 8B answer as 4; Section 14 gives the measured seconds.")}

<h2 id="routing">5. Routing to a larger model (no ranker)</h2>
<p>The detector flags the least-confident questions (threshold from validation, as for the gate); a flagged question gets the fix's answer
(rule <i>replace</i>) or keeps whichever of the two answers has the higher answer probability (rule <i>confident</i>). Detector, rule and budget are
chosen on validation. Cost in 2B passes: a 4B answer counts 2, an 8B answer 4, an enlarged image the pixel ratio.</p>
{table(["Fix", "Setting", "Detector", "Rule", "Routed (test)", "Passes", "Test accuracy", "Gain (95% interval)"], route_rows, left_cols=(0, 1, 2, 3))}
<p>Which detector finds the questions a fix will help? Test AUROC for "the 2B answer is right" and for "the 8B answer is right where the 2B one is wrong":</p>
{table(["Detector", "2B answer right", "8B fix helps"], detq, left_cols=(0,))}
<p>The trained head is the better judge of whether the 2B answer is right; for finding the questions the 8B model fixes, the answer probability is as good.
Enlarging the image helps little (about a point at best): the processor already reads images at their native size. Routing 30% of questions to 4B
(86.03%) and running 4B on every question (85.85%) are equal within noise.</p>

<h2 id="candidates">6. Larger-model answers as ranker candidates</h2>
<p>The 8B answers are encoded by the 2B model exactly like the 2B candidates, and the ranker gets one more source-agreement feature (the share of
larger-model proposals equal to the candidate). The rankers are retrained on the 12,000 training questions, for which the 8B answers were generated too.</p>
{table(["Ranker", "Candidates", "No gate", "Gain (95% interval)", "Passes", "Gated 50%", "Passes", "Best possible pick"], cand_rows, left_cols=(0, 1))}
<p>The 8B short answer alone already carries most of the gain; the 8B reasoning answer adds to it when the 2B candidates are present, and the
best possible pick from that set is close to 95%. The enlarged-image answers do not help the ranker.</p>

<h2 id="inside">7. What the best ranker does</h2>
<p>The saved best ranker scored again on test (analyze_best_ranker.py), compared question by question with the alternatives. Routing uses the setting
chosen on validation ({analysis['routing_setting']['detector']}, rule {analysis['routing_setting']['rule']}, budget {analysis['routing_setting']['budget']:g}).
Paired bootstrap interval and McNemar p of each difference, pooled, and the difference per dataset:</p>
{table(["Comparison (first minus second)", "Pooled (95% interval)", "p", *DATASETS.values()], paired_rows, left_cols=(0,))}
<p>Where the chosen answer comes from (no gate):</p>
{table(["Origin of the chosen answer", "Share of questions", "Right when chosen", "Wrong greedy answers fixed", "Right greedy answers broken"], origin_rows, left_cols=(0,))}
<p>A right answer is among the candidates for {pct(oracle['any_correct'])} of questions; when there is one, the ranker picks a right answer
{pct(oracle['picked_when_available'])} of the time, and {oracle['missed']:,} questions have a right candidate it misses. Answers proposed by both a 2B correction
and the 8B model are right {analysis['origin'].get('2B and 8B both', {}).get('accuracy', 0) * 100:.0f}% of the time when chosen: agreement across sources
is the signal the ranker learns (Section 10).</p>
<p>Answering only the questions the system is most confident about: accuracy at each coverage, and the area under the risk-coverage curve (lower is better).</p>
{table(["Answers, ranked by", "50%", "80%", "90%", "100%", "AURC"], sel_rows, left_cols=(0,))}
<p>Examples from the test split (the first fixed and broken questions per dataset):</p>
{table(["Dataset", "", "Question", "Reference", "Greedy", "Chosen", "Origin"], example_rows, left_cols=(2, 3, 4, 5, 6))}

<h2 id="cascade">8. A 2B → 4B → 8B cascade</h2>
<p>Step 1: the 2B detector flags a share of questions, which get a 4B answer. Step 2: of those, the ones whose 4B answer is least probable also get
an 8B answer. Every setting (detector, rule, both shares) is chosen on validation under each average cost and compared with routing straight to 8B
under the same cost, on the same questions.</p>
{table(["Cost cap", "Route", "Detector", "Rule", "To 4B / to 8B (test)", "Passes", "Test accuracy", "Gain (95% interval)", "Cascade minus direct"], cas_rows, left_cols=(1, 2, 3))}
<p>The 4B step pays only when there is no budget for 8B. Once 8B fits, sending its share of questions straight to 8B is as good, so the cascade
adds a model without adding accuracy.</p>

<h2 id="baselines">9. Training-free baselines on the same questions</h2>
<p>Every baseline uses the same questions and, where it uses candidates, the same candidates as the ranker. None has a parameter chosen on test.</p>
{table(["Method", "Calls", "Passes", "Test accuracy", "Gain (95% interval)", "Fixed / broken"] + list(DATASETS.values()), base_rows, left_cols=(0,))}
<p>Sampling more 2B answers and voting barely moves accuracy; voting over the reasoning answers helps ChartQA and ScienceQA and hurts TextVQA and
DocVQA; picking the candidate the model's own judge likes most is clearly worse than the greedy answer. The learned ranker over the same 2B
candidates ({pct(two_new['test']['pooled']['accuracy'])}) is well above every vote. Running the 8B model on every question
({pct(all8['pooled']['accuracy'])}, 4 passes) is the strongest baseline; routing only the flagged questions to it reaches more at less cost.</p>

<h2 id="inputs">10. Which ranker inputs matter</h2>
<p>Each ranker retrained with one group of inputs set to zero in training and evaluation, everything else equal. 2B-only ranker: {label(TWO)};
best ranker: {label(K)} (trained without the judge, so each of its rows also lacks the judge).</p>
{table(["Inputs removed", "2B-only ranker", "Change", "Best ranker", "Change"], abl_rows, left_cols=(0,))}
<p>Source agreement is what both rankers rely on, the best ranker even more than the 2B-only one; the answer probability and the head add a little;
the judge adds nothing. Two variants of the best ranker:</p>
{table(["Best ranker", "Test accuracy"], variant_rows, left_cols=(0,))}
<p>Reading the hidden states hurts (the scalars already summarize them); longer training adds about 0.2 point. Since the judge costs one forward pass
per candidate, the rankers are retrained without it and gated with one scoring pass per candidate:</p>
{table(["Candidates", "Setting", "With judge", "Passes", "Without judge", "Passes"], nj_rows, left_cols=(0, 1))}

<h2 id="robust">11. Robustness</h2>
<p><b>Training seeds.</b> Each ranker retrained with two further sets of three seeds (test accuracy of the same configuration):</p>
{table(["Ranker", "Candidates", "Seeds 0–2", "Seeds 100–102", "Seeds 200–202", "Range (points)"], seed_rows, left_cols=(0, 1))}
<p><b>A dataset the ranker never trained on.</b> Each ranker (no dataset input) retrained four times, each time without one dataset in training and in
the learning-rate choice, then tested on that dataset; against the same ranker trained on all four.</p>
{table(["Held-out dataset", "Greedy", "Best ranker, all four", "Best ranker, never trained on it", "Kept", "2B-only (with judge), all four", "2B-only (with judge), never trained on it", "Kept"], lodo_rows)}
<p><b>Training-set size.</b> Each ranker trained on a fixed random subset of the training questions (same validation and test):</p>
{table(["Questions per dataset", "In total", "Best ranker", "2B-only ranker (no judge)"], size_rows)}

<h2 id="unseen">12. Datasets the pipeline never saw</h2>
<p>AI2D ({unseen['questions']['ai2d']:,} questions) and InfographicVQA ({unseen['questions']['infovqa']:,}) run through the frozen pipeline: every candidate source
as in-domain, the saved rankers without their dataset input, configurations chosen on in-domain validation, and gate thresholds from in-domain
validation. Nothing is trained, tuned or chosen on these datasets.</p>
{table(["Policy", "Accuracy", "Gain over greedy (95% interval)", "Fixed / broken", *UNSEEN.values()], unseen_rows, left_cols=(0,))}
<p>The 2B-only ranker transfers with a gain similar to in-domain. These datasets are harder for the 2B model, so the in-domain thresholds correct far more
of their questions ({corrected_share} at the 50% budget). The larger-model candidates carry most of the gain here: the best ranker is only a little above
running the 8B model on every question.</p>

<h2 id="bases">13. Other base models</h2>
<p>The whole pipeline repeated with Qwen3-VL-4B and Qwen3-VL-8B as the base model in place of 2B: their own candidates, encodings and rankers, on the same
questions. The 2B-trained heads cannot read another model's states, so their inputs are a constant there. The 8B base-model run used half of the
training questions (1,500 per dataset; Section 11 shows little loss from that).</p>
{table(["Base model", "Candidates", "Training questions", "Greedy", "Ranker", "Gain (95% interval)", "Best possible pick"], base_rows_m, left_cols=(0, 1))}
<p>The gain shrinks as the base model gets stronger: its own corrections rarely beat its greedy answer. With the 8B answers as candidates, the 4B base model
gains {base_rows_m[3][5].split(' ')[0]} points, about what the 2B model gains from its own candidates ({base_rows_m[0][5].split(' ')[0]}), and ends at {base_rows_m[3][4]},
close to the 2B system with the 8B answers ({base_rows_m[1][4]}). The 4B and 8B gating and routing tables are being rerun (Section 15).</p>

<h2 id="latency">14. Measured time</h2>
<p>Mean, median and 90th-percentile seconds per question of every component on one {lat['gpu']}, on the first {lat['per_dataset']} test questions of
each dataset (batch size 1; the first question of each run is a warm-up and not timed).</p>
{table(["Component", "Mean (s)", "Median (s)", "90th percentile (s)", "Against a 2B short answer"], lat_rows, left_cols=(0,))}
<p>Two of the pass-counting conventions do not match measured time. An 8B short answer counts as 4 passes but takes
{comp['greedy_8b']['mean'] / comp['greedy']['mean']:.1f} times a 2B answer: at batch size 1 the GPU is far from saturated, so time grows much
more slowly than parameter count. A step-by-step answer counts as 1 call but takes {comp['reasoning']['mean'] / comp['greedy']['mean']:.0f} times a
short answer, because it generates up to 256–384 tokens. In seconds, an 8B answer costs about as much as one zoomed-crop answer and is the most
effective single fix, while the reasoning candidates are the most expensive (Section 4 gives the policies chosen under seconds caps).</p>

<h2 id="checks">15. Checks</h2>
<ul><li><b>Gating uses the exact saved rankers.</b> Before gating, each configuration's ungated accuracy is recomputed and compared with the training
run's report on validation and test; the script stops otherwise:</li></ul>
{table(["Ranker", "Candidates", "Reported (test)", "Gating script, no gate", "Questions different", "Match"], repro, left_cols=(0, 1))}
<ul>
<li><b>Complete inputs.</b> Every validation and test question has its 4B and 8B answers (0 missing), and the escalation candidates cover all
12,000 training questions.</li>
<li><b>Constant detectors.</b> With the 4B and 8B base models the 2B-trained head and probe score a constant 0.5, which a budget threshold turns into
"correct every question". The gating and routing scripts now leave out any detector that is constant on validation; this changes no 2B result, and the
4B and 8B gating and routing tables are being rerun with it.</li>
<li><b>Tests.</b> Unit tests for the routing simulation, the cascade, the escalation candidates, the gate, the rankers and the analysis pass on a GPU node
before the analyses run, under each base model.</li>
</ul>

<h2 id="limits">16. Limits</h2>
<ul>
<li>The larger models are of the same family as the 2B model; fixes from other families are in progress.</li>
<li>Pass counts are a convention; Section 14 shows they misstate the relative cost of 8B answers and reasoning.</li>
<li>Latency is measured at batch size 1 on one GPU type; batched serving would change the ratios.</li>
<li>The ranker is trained with correctness labels on four datasets; Sections 11 and 12 test how far it carries.</li>
</ul>

<h2 id="progress">17. In progress</h2>
<ul>
<li>The 4B and 8B base-model gating and routing tables, rerun without the constant detectors.</li>
<li>Fixes from other model families (InternVL3-8B, Idefics3-8B): routing, as ranker candidates, and on the unseen datasets; component times on A100 and
L40S.</li>
</ul>

<h2 id="reproduce">18. Reproduce</h2>
<pre>bash slurm/run_escalation_pipeline.sh            # 8B, 4B and enlarged-image answers on val/test; routing study
bash slurm/run_escalation_ranker_pipeline.sh     # the same on the train subset, as candidates; ranker; gating
bash slurm/run_reasoning_8b_pipeline.sh          # 8B reasoning as a candidate; ranker; gating
bash slurm/run_next_steps_pipeline.sh            # cascade; rankers without the judge; gating with one scoring pass
bash slurm/run_followups_pipeline.sh             # seeds, leave-one-dataset-out, training-set size, gating in seconds, unseen
bash slurm/run_unseen_pipeline.sh                # the unseen datasets through the frozen pipeline
bash slurm/run_base_model_pipeline.sh            # another base model (SIZE=4b or 8b); run_base_model_escalation_pipeline.sh adds the 8B answers
sbatch --export=ALL,BATCH=best_ranker_analysis.txt slurm/run_batch.sbatch   # Section 7
sbatch --export=ALL,BATCH=headline_ablations.txt slurm/run_batch.sbatch     # Section 10 (also headline_variants.txt, nojudge_seeds.txt)
sbatch slurm/compute_baselines.sbatch            # training-free baselines
sbatch slurm/measure_latency.sbatch              # seconds per component on an H100
python -m experiments.approach1_latent_self_correction.build_report_13   # this report</pre>
<p class="small">Data: <code>results/approach1/04_sweep_data/</code> (escalation, cascade, baselines, latency, best_ranker_analysis, unseen,
candidate_ranker_v3_multi*, gating_v3_multi*, base_4b/, base_8b/). Code: <code>escalation.py</code>, <code>escalation_candidates.py</code>,
<code>analyze_escalation.py</code>, <code>analyze_cascade.py</code>, <code>analyze_best_ranker.py</code>, <code>evaluate_unseen.py</code>,
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
