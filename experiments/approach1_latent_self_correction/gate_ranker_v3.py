# gate_ranker_v3.py
#
# Gating for the v3 correction step (train_ranker_v3.py). Without a gate every question gets every
# candidate source and the ranker; with one, only the questions a detector flags are corrected and the
# rest keep the 2B greedy answer at the cost of one generation.
#
# Detectors (low = likely wrong = corrected first), all of the greedy answer or the question:
#   token       token confidence of the greedy answer, from its generation (free)
#   pre_head    pre-generation score of the unified head on the cached question embedding (free)
#   after_head  head score of the greedy answer (one encoding pass of the greedy answer)
#   judge       the model's own P(Yes) that the greedy answer is right (one forward pass)
#   random      each question corrected with probability b (the baseline to beat)
# Thresholds come from validation (gating.py); on test each question is gated on its own score.
#
# Cost is counted per question from what it actually needed. Generation calls use train_ranker_v3's
# convention (greedy 1, each sample 1, each correction source 1, the grounded crops one shared box call
# plus one each, each sampled reasoning answer 1); passes add two scoring passes (encoding and judge)
# per distinct candidate of a corrected question, and the detector's pass when it needs one.
# Seconds: with results/latency.json (measure_latency.py, one H100) each question also gets an estimated
# wall-clock cost, the sum of the measured mean seconds of the components it runs (SECONDS_OF below), and
# the headlines add the best setting under 0.3, 0.5, 1 and 2 seconds per question (chosen on validation).
# --drop gates a ranker trained with train_ranker_v3.py --drop (the same input columns are zeroed here);
# --scoring-passes sets the scoring passes per candidate: 1 for a ranker trained without the judge
# (--drop judge), which needs only the encoding pass.
#
# Models: the trained rankers exactly as saved by train_ranker_v3.py (--models), so the ungated row of
# every configuration must reproduce the accuracy that run reported (--reported); the script records
# the check and stops if it fails. With --with-cot-samples the sampled reasoning answers of
# correction_v3_cot.py are merged and the *_multi configurations are gated as well.
#
# Configurations: the validation choice of report 12 (T1_k5_everything), the cost-limited choice
# (T1_k0_ground_cot), reasoning only (T1_k0_cot) and crops + reasoning (T1_k0_zoom_cot), plus
# T1_k0_cot_multi and T1_k0_ground_cot_multi with sampled reasoning. Headlines, each chosen on
# validation and read once on test:
#   per configuration   the best detector at budgets 20, 30 and 50%; the smallest budget whose
#                       validation gain keeps 90% of correcting everything
#   overall             the best (configuration, detector, budget) under an average cost of 2, 3 and 5
#                       passes per question
# Writes results/gating_v3<tag>.json and .md.

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.correction_stats import paired_difference, policy_row
from experiments.approach1_latent_self_correction.gating import BUDGETS, combine, gate, random_gate, threshold
from experiments.approach1_latent_self_correction.train_candidate_ranker import load_questions
from experiments.approach1_latent_self_correction.train_ranker_v2 import RESULTS, merge_v2
from experiments.approach1_latent_self_correction.train_ranker_v3 import (
    DROPPED,
    FEATURE_GROUPS,
    GROUPS,
    MULTI_SETS,
    V4_DIR,
    V5_DIR,
    enable_escalation,
    SETS,
    RankerV3,
    candidate_set,
    choose,
    generation_calls,
    merge_cot_samples,
)

CONFIGS = ("T1_k5_everything", "T1_k0_ground_cot", "T1_k0_cot", "T1_k0_zoom_cot")
MULTI_CONFIGS = ("T1_k0_cot_multi", "T1_k0_ground_cot_multi")
ESCALATION_CONFIGS = ("T1_k0_esc8b", "T1_k0_ground_cot_multi_esc8b", "T1_k0_ground_cot_multi_esc", "T1_k0_everything_multi_esc")
REASONING_8B_CONFIGS = ("T1_k0_cot8b", "T1_k0_ground_cot_multi_cot8b", "T1_k0_ground_cot_multi_esc8b_cot8b")
DETECTORS = ("token", "pre_head", "after_head", "judge", "random")
DETECTOR_PASS = {"token": 0, "pre_head": 0, "after_head": 1, "judge": 1, "random": 0}
FIXED_BUDGETS = (0.2, 0.3, 0.5)
COST_CAPS = (2.0, 3.0, 5.0)
KEEP = 0.9
RANDOM_DRAWS = 50
VARIANT = "scalars_ds"
MAX_DIFFERENT_QUESTIONS = 3  # allowance for GPU non-determinism flipping a near-tie
SCORING = {"passes": 2}  # scoring passes per distinct candidate (encoding + judge); set by --scoring-passes
SECONDS_CAPS = (0.3, 0.5, 1.0, 2.0)
LATENCY: dict = {}  # component -> mean seconds, from results/latency.json (empty: no seconds column)
# Which measured component each candidate source costs. The upscaled-image answers were not timed; they are
# estimated as a 2B short answer times the pixel ratio (scale squared), the same rule as their pass count.
SECONDS_OF = {"cot": ("reasoning", 1.0), "cot_s0": ("reasoning_x3", 1 / 3), "cot_s1": ("reasoning_x3", 1 / 3), "cot_s2": ("reasoning_x3", 1 / 3),
              "vcd_0.5": ("vcd", 1.0), "vcd_1": ("vcd", 1.0), "zoom_0": ("zoom", 1.0), "zoom_1": ("zoom", 1.0), "zoom_2": ("zoom", 1.0),
              "zoom_3": ("zoom", 1.0), "esc_8b": ("greedy_8b", 1.0), "cot_8b": ("reasoning_8b", 1.0),
              "up1.5": ("greedy", 2.25), "up2": ("greedy", 4.0)}
DETECTOR_SECONDS = {"token": None, "pre_head": None, "after_head": "encode", "judge": "judge", "random": None}


def generation_seconds(k: int, sources: tuple) -> float:
    """Estimated generation seconds of a corrected question: greedy, k samples, the grounding calls once, each source."""
    total = LATENCY["greedy"] + LATENCY["samples_10"] * k / 10
    if any(s in ("ground", "ground_ctx") for s in sources):
        total += LATENCY["grounding"]
    for s in sources:
        if s in SECONDS_OF:
            component, factor = SECONDS_OF[s]
            total += LATENCY[component] * factor
    return total


def scoring_seconds() -> float:
    """Seconds to score one distinct candidate: the encoding pass, plus the judge pass when the ranker reads it."""
    return LATENCY["encode"] + (LATENCY["judge"] if SCORING["passes"] == 2 else 0.0)


def base_seconds(detector: str) -> float:
    """Seconds of an uncorrected question: its greedy answer and the detector's pass, if any."""
    extra = DETECTOR_SECONDS[detector]
    return LATENCY.get("greedy", 0.0) + (LATENCY.get(extra, 0.0) if extra else 0.0)


def parse(key: str) -> tuple[str, int, str]:
    """('1', 5, 'everything') from 'T1_k5_everything'."""
    temperature, rest = key[1:].split("_k", 1)
    k, name = rest.split("_", 1)
    return temperature, int(k), name


def attach_gates(questions: list[dict], split: str) -> None:
    """Detector scores of every question: from the T=1 pools (token, pre_head) and the encoded greedy answer."""
    pools = {}
    for path in sorted((RESULTS / "pools" / "T1" / split).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if "greedy" in record:
                    pools[(record["dataset"], record["index"])] = record
    for q in questions:
        record, greedy = pools[q["key"]], q["encoded"][q["greedy"]]
        q["gates"] = {"token": record["greedy"]["token_confidence"], "pre_head": record["pre_generation"]["head"],
                      "after_head": greedy["head_confidence"], "judge": q["judge"].get(q["greedy"], 0.5)}


def load_models(path: Path, variant: str, device) -> list[RankerV3]:
    """The three seeds of one v3 variant, as saved by train_ranker_v3.py."""
    saved = torch.load(path, map_location=device, weights_only=False)[variant]
    models = []
    for state in saved:
        model = RankerV3(variant, state["network.0.weight"].shape[0]).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def split_table(questions: list[dict], models: list, device, configs: tuple) -> dict:
    """Per question: dataset, greedy and corrected correctness per configuration, cost, detector scores."""
    table = {"dataset": np.array([q["key"][0] for q in questions]),
             "greedy": np.array([bool(q["encoded"][q["greedy"]]["correct"]) for q in questions]),
             "scores": {d: np.array([q["gates"][d] for q in questions], dtype=float) for d in DETECTORS if d != "random"},
             "corrected": {}, "passes": {}, "calls": {}}
    for key in configs:
        temperature, k, name = parse(key)
        config = (temperature, k, SETS[name])
        table["corrected"][key] = np.array(choose(models, questions, config, device))
        candidates = np.array([len(candidate_set(q, temperature, k, SETS[name])[0]) for q in questions])
        table["calls"][key] = generation_calls(k, SETS[name])
        table["passes"][key] = table["calls"][key] + SCORING["passes"] * candidates
        if LATENCY:
            table.setdefault("seconds", {})[key] = generation_seconds(k, SETS[name]) + scoring_seconds() * candidates
    return table


def routed(table: dict, detector: str, budget: float, reference: dict, seed: int = 1) -> np.ndarray:
    """Questions sent to correction under one detector and budget, thresholds from `reference` (validation)."""
    if detector == "random":
        return random_gate(len(table["greedy"]), budget, seed)
    return gate(table["scores"][detector], threshold(reference["scores"][detector], budget))


def evaluate(table: dict, key: str, detector: str, budget: float, reference: dict, seed: int = 1) -> dict:
    """Outcome, share corrected and average cost of one gated setting."""
    mask = routed(table, detector, budget, reference, seed)
    final = combine(mask, table["corrected"][key], table["greedy"])
    base = 1 + DETECTOR_PASS[detector]
    passes = np.where(mask, table["passes"][key], base)
    calls = np.where(mask, table["calls"][key], 1)
    seconds = np.where(mask, table["seconds"][key], base_seconds(detector)) if "seconds" in table else np.zeros(len(mask))
    return {"final": final, "share": float(mask.mean()), "passes": float(passes.mean()), "calls": float(calls.mean()),
            "seconds": float(seconds.mean())}


def summary(table: dict, key: str, detector: str, budget: float, reference: dict) -> dict:
    """Accuracy, share and costs; the random detector averaged over draws."""
    if detector != "random":
        e = evaluate(table, key, detector, budget, reference)
        return {"accuracy": float(e["final"].mean()), "share": e["share"], "passes": e["passes"], "calls": e["calls"], "seconds": e["seconds"]}
    draws = [evaluate(table, key, "random", budget, reference, s) for s in range(RANDOM_DRAWS)]
    return {m: float(np.mean([d["final"].mean() if m == "accuracy" else d[m] for d in draws])) for m in ("accuracy", "share", "passes", "calls", "seconds")}


def grid(table: dict, reference: dict) -> dict:
    """Every configuration, detector and budget."""
    return {key: {d: [summary(table, key, d, b, reference) for b in BUDGETS] for d in DETECTORS} for key in table["corrected"]}


def breakdown(table: dict, final: np.ndarray) -> dict:
    """Paired comparison with greedy, pooled and per dataset."""
    rows = {"pooled": policy_row(table["greedy"].tolist(), final.tolist())}
    for d in DATASET_NAMES:
        m = table["dataset"] == d
        rows[d] = policy_row(table["greedy"][m].tolist(), final[m].tolist())
    return rows


def read_test(val: dict, test: dict, val_grid: dict, name: str, key: str, detector: str, budget: float) -> dict:
    """One setting fixed on validation, read once on test, with the random gate at the same budget."""
    e = evaluate(test, key, detector, budget, val)
    rand = evaluate(test, key, "random", budget, val)
    v = val_grid[key][detector][BUDGETS.index(budget)]
    return {"name": name, "configuration": key, "detector": detector, "budget": budget,
            "val": {"accuracy": v["accuracy"], "passes": v["passes"], "share": v["share"]},
            "test_share": e["share"], "test_passes": e["passes"], "test_calls": e["calls"], "test_seconds": e["seconds"],
            "test": breakdown(test, e["final"]),
            "test_random_accuracy": summary(test, key, "random", budget, val)["accuracy"],
            "vs_random": paired_difference(rand["final"].tolist(), e["final"].tolist())}


def best_detector(val_grid: dict, key: str, budget: float) -> str:
    """Detector with the best validation accuracy at one budget (random excluded)."""
    i = BUDGETS.index(budget)
    return max((d for d in DETECTORS if d != "random"), key=lambda d: val_grid[key][d][i]["accuracy"])


def headlines(val: dict, test: dict, val_grid: dict) -> list[dict]:
    """Every validation-chosen setting, read once on test."""
    out = []
    base = float(val["greedy"].mean())
    for key in val["corrected"]:
        out.append(read_test(val, test, val_grid, "correct every question (no gate)", key, "token", 1.0))
        for b in FIXED_BUDGETS:
            out.append(read_test(val, test, val_grid, f"best detector at a {int(b * 100)}% budget", key, best_detector(val_grid, key, b), b))
        full = val_grid[key]["token"][-1]["accuracy"]
        target = base + KEEP * (full - base)
        b = next(b for b in BUDGETS if val_grid[key][best_detector(val_grid, key, b)][BUDGETS.index(b)]["accuracy"] >= target)
        out.append(read_test(val, test, val_grid, f"smallest budget keeping {int(KEEP * 100)}% of the full validation gain", key,
                             best_detector(val_grid, key, b), b))
    for cap in COST_CAPS:
        options = [(key, d, b) for key in val["corrected"] for d in DETECTORS if d != "random" for b in BUDGETS
                   if val_grid[key][d][BUDGETS.index(b)]["passes"] <= cap]
        key, d, b = max(options, key=lambda o: (val_grid[o[0]][o[1]][BUDGETS.index(o[2])]["accuracy"], -o[2]))
        out.append(read_test(val, test, val_grid, f"best under {cap:g} passes per question (any configuration)", key, d, b))
    for cap in (SECONDS_CAPS if "seconds" in val else ()):
        options = [(key, d, b) for key in val["corrected"] for d in DETECTORS if d != "random" for b in BUDGETS
                   if val_grid[key][d][BUDGETS.index(b)]["seconds"] <= cap]
        if options:
            key, d, b = max(options, key=lambda o: (val_grid[o[0]][o[1]][BUDGETS.index(o[2])]["accuracy"], -o[2]))
            out.append(read_test(val, test, val_grid, f"best under {cap:g} s per question on one H100 (any configuration)", key, d, b))
    return out


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.2f}"


def markdown(result: dict) -> str:
    """Headline table and per-dataset rows of the gated settings."""
    lines = ["# Gating the v3 correction step", "",
             "Thresholds on the detector score come from validation; on test each question is gated on its own score. "
             f"Ranker: v3 {result['variant']} from `{result['models']}`. Greedy on test: {pct(result['greedy_test'])}. Passes: model passes per question "
             f"(generation calls plus {result.get('scoring_passes', 2)} scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only."
             + (f" Ranker inputs zeroed: {', '.join(result['dropped'])}." if result.get("dropped") else ""), "",
             "Reproduction check (ungated test accuracy against the training run's report): "
             + ", ".join(f"{k} {pct(v['ours'])} vs {pct(v['reported'])}" for k, v in result["reproduction"].items()), "",
             "Seconds: estimated wall-clock seconds per question on one H100, from the measured component times (results/latency.json)." if result.get("latency") else "",
             "",
             "| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Seconds | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |",
             "|---|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for h in result["headlines"]:
        t, v = h["test"]["pooled"], h["vs_random"]
        lines.append(f"| {h['name']} | {h['configuration']} | {h['detector']} | {int(round(100 * h['budget']))}% / {100 * h['test_share']:.1f}% | "
                     f"{h['test_passes']:.2f} | {h['test_calls']:.2f} | {h.get('test_seconds', 0):.2f} | {pct(t['accuracy'])} | {pts(t['difference'])} ({pts(t['ci_low'])} to {pts(t['ci_high'])}) | "
                     f"+{t['fixed']} / -{t['broken']} | {pts(v['difference'])} ({pts(v['ci_low'])} to {pts(v['ci_high'])}) |")
    lines += ["", "Per dataset (gain over greedy on test):", "", "| Setting | Configuration | " + " | ".join(DATASET_NAMES) + " |",
              "|---|---|" + "---|" * len(DATASET_NAMES)]
    for h in result["headlines"]:
        lines.append(f"| {h['name']} | {h['configuration']} | " + " | ".join(
            f"{pts(h['test'][d]['difference'])} ({pts(h['test'][d]['ci_low'])} to {pts(h['test'][d]['ci_high'])})" for d in DATASET_NAMES) + " |")
    return "\n".join(lines) + "\n"


def reproduction(tables: dict, reported_path: Path, variant: str) -> dict:
    """Ungated accuracy of each configuration on val and test against the training run's report."""
    reported = json.loads(reported_path.read_text())
    out = {}
    for key in tables["test"]["corrected"]:
        out[key] = {"ours": float(tables["test"]["corrected"][key].mean()),
                    "reported": reported["test"][key][f"ranker_v3_{variant}"]["pooled"]["accuracy"],
                    "ours_val": float(tables["val"]["corrected"][key].mean()),
                    "reported_val": reported["val"][key][f"ranker_v3_{variant}"]["pooled"]["accuracy"]}
        n_test, n_val = len(tables["test"]["greedy"]), len(tables["val"]["greedy"])
        out[key]["questions_different_test"] = round(abs(out[key]["ours"] - out[key]["reported"]) * n_test)
        out[key]["questions_different_val"] = round(abs(out[key]["ours_val"] - out[key]["reported_val"]) * n_val)
        out[key]["match"] = max(out[key]["questions_different_test"], out[key]["questions_different_val"]) <= MAX_DIFFERENT_QUESTIONS
    return out


def main() -> None:
    """Score every configuration once per split, check it against the training run, then gate."""
    parser = argparse.ArgumentParser(description="Gating for the v3 correction step.")
    parser.add_argument("--models", type=Path, default=RESULTS / "candidate_ranker_v3_models.pt")
    parser.add_argument("--reported", type=Path, default=RESULTS / "candidate_ranker_v3.json",
                        help="The training run's results, which the ungated rows must reproduce.")
    parser.add_argument("--variant", default=VARIANT)
    parser.add_argument("--with-cot-samples", action="store_true", help="Merge correction_v3 and gate the *_multi configurations too.")
    parser.add_argument("--with-escalation", action="store_true",
                        help="Merge correction_v4 (8B and upscaled answers) and gate the escalation configurations; implies --with-cot-samples.")
    parser.add_argument("--with-reasoning-8b", action="store_true", help="Merge correction_v5 (8B reasoning) too; implies --with-escalation.")
    parser.add_argument("--tag", default="")
    parser.add_argument("--drop", nargs="+", default=[], choices=sorted(FEATURE_GROUPS),
                        help="The input groups the ranker was trained without (train_ranker_v3.py --drop).")
    parser.add_argument("--latency", type=Path, default=RESULTS / "latency.json", help="Measured component seconds (measure_latency.py); skipped if absent.")
    parser.add_argument("--scoring-passes", type=int, default=2, choices=(1, 2),
                        help="Scoring passes per distinct candidate: 2 (encoding + judge) or 1 (encoding only, for --drop judge).")
    args = parser.parse_args()
    if args.scoring_passes == 1 and "judge" not in args.drop:
        raise SystemExit("--scoring-passes 1 needs a ranker trained without the judge (--drop judge)")
    SCORING["passes"] = args.scoring_passes
    if args.latency.exists():
        LATENCY.update({name: c["mean"] for name, c in json.loads(args.latency.read_text())["components"].items()})
    output = RESULTS / f"gating_v3{args.tag}.json"
    configs = CONFIGS
    args.with_escalation = args.with_escalation or args.with_reasoning_8b
    args.with_cot_samples = args.with_cot_samples or args.with_escalation
    if args.with_cot_samples:
        SETS.update(MULTI_SETS)
        configs = CONFIGS + MULTI_CONFIGS
    if args.with_escalation:
        enable_escalation()
        configs = configs + ESCALATION_CONFIGS + (REASONING_8B_CONFIGS if args.with_reasoning_8b else ())
    for group in args.drop:  # the same columns train_ranker_v3.py zeroed (after enable_escalation, as in training)
        DROPPED.update(FEATURE_GROUPS[group])
        if group == "agreement":
            DROPPED.update(f"from_{g}" for g in GROUPS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args.models, args.variant, device)
    tables = {}
    for split in ("val", "test"):
        questions = merge_v2(load_questions(split), split)
        if args.with_cot_samples:
            print(f"[cot samples] {split}: {merge_cot_samples(questions, split)} questions", flush=True)
        if args.with_escalation:
            print(f"[escalation] {split}: {merge_cot_samples(questions, split, V4_DIR)} questions", flush=True)
        if args.with_reasoning_8b:
            print(f"[8B reasoning] {split}: {merge_cot_samples(questions, split, V5_DIR)} questions", flush=True)
        attach_gates(questions, split)
        tables[split] = split_table(questions, models, device, configs)
        print(f"[scored] {split}: {len(questions)} questions", flush=True)
    check = reproduction(tables, args.reported, args.variant)
    for key, row in check.items():
        print(f"[reproduction] {key}: test {row['ours']:.6f} vs {row['reported']:.6f}, val {row['ours_val']:.6f} vs {row['reported_val']:.6f}"
              f" -> {'ok' if row['match'] else 'MISMATCH'}", flush=True)
    if not all(row["match"] for row in check.values()):
        raise SystemExit("the ungated rows do not reproduce the training run; not gating")
    val, test = tables["val"], tables["test"]
    val_grid, test_grid = grid(val, val), grid(test, val)
    result = {"models": args.models.name, "variant": args.variant, "reproduction": check,
              "scoring_passes": args.scoring_passes, "dropped": sorted(args.drop), "latency": dict(LATENCY),
              "budgets": list(BUDGETS), "configurations": list(configs), "detectors": list(DETECTORS),
              "greedy_val": float(val["greedy"].mean()), "greedy_test": float(test["greedy"].mean()),
              "ungated_test_accuracy": {k: float(test["corrected"][k].mean()) for k in configs},
              "headlines": headlines(val, test, val_grid), "curves": {"val": val_grid, "test": test_grid}}
    output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"[done] wrote {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
