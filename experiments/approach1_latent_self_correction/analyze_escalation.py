# analyze_escalation.py
#
# The routing study of report 13: keep our detector, swap the fix. For every validation and test
# question we have the 2B greedy answer with its detector scores (results/pools/T1) and the answers of
# each fix from escalation.py (Qwen3-VL-8B, and 2B on the image enlarged 1.5x and 2x). This script
# simulates sending the least-confident fraction of questions to a fix, for every detector and budget,
# without new generation.
#
#   detectors   pre_head / pre_probe     score of the question before any answer (constant question
#                                        embedding, pre-generation mode of the unified head / probe)
#               after_head / after_probe score of the 2B greedy answer (joint mode)
#               token                    token confidence of the 2B greedy answer (the usual baseline)
#               random                   the same budget sent at random (expected value over 50 draws)
#   rules       replace                  a routed question takes the fix's answer
#               confident                it takes the fix's answer only if its token confidence is higher
#   budgets     0, 5, ..., 100% of questions. A budget is turned into a fixed threshold on the detector
#               score from VALIDATION (gating.py); on test each question is routed on its own score, so
#               the share actually routed on test is measured, not imposed.
#
# Everything is chosen on validation and test is read once per headline: the best detector and rule at
# a 20% and a 30% budget, the best budget overall, and the smallest budget that keeps 80% of that gain.
# Writes results/escalation.json and results/escalation.md.

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.correction_stats import auroc, paired_difference, policy_row
from experiments.approach1_latent_self_correction.gating import BUDGETS, gate, random_gate, threshold

HERE = Path(__file__).parent
RESULTS = HERE / "results"
POOLS = RESULTS / "pools" / "T1"
ESCALATION = RESULTS / "escalation"
DETECTORS = ("pre_head", "pre_probe", "after_head", "after_probe", "token", "random")
RULES = ("replace", "confident")
# Approximate cost of one fix answer in 2B greedy passes: 8B has about 4x the parameters; an image
# enlarged by s has about s^2 times the image tokens, which dominate the cost of a short answer.
FIX_COST = {"8b": 4.0, "up1.5": 2.25, "up2": 4.0}
RANDOM_DRAWS = 50


def read_jsonl(paths) -> list[dict]:
    """Every record of a set of JSONL files."""
    rows = []
    for path in paths:
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def load_greedy(split: str) -> dict[tuple, dict]:
    """The 2B greedy answer and detector scores of every question, by (dataset, index)."""
    out = {}
    for r in read_jsonl(sorted((POOLS / split).glob("*.jsonl"))):
        if "greedy" not in r:
            continue
        g = r["greedy"]
        out[(r["dataset"], r["index"])] = {
            "correct": bool(g["correct"]), "token": g["token_confidence"],
            "after_head": g["head_confidence"], "after_probe": g["probe_confidence"],
            "pre_head": r["pre_generation"]["head"], "pre_probe": r["pre_generation"]["probe"],
        }
    return out


def load_fixes(split: str) -> dict[str, dict[tuple, dict]]:
    """The answers of each fix by (dataset, index); records with an error are left out."""
    fixes = {}
    for folder in sorted(ESCALATION.glob("model_*")):
        rows = read_jsonl(sorted((folder / split).glob("*.jsonl")))
        fixes[folder.name.split("_", 1)[1]] = {(r["dataset"], r["index"]): r["answer"] for r in rows if "answer" in r}
    rows = read_jsonl(sorted((ESCALATION / "upscale" / split).glob("*.jsonl")))
    for r in rows:
        for scale, result in r.get("upscaled", {}).items():
            fixes.setdefault(f"up{scale}", {})[(r["dataset"], r["index"])] = result
    return fixes


def table(greedy: dict, fix: dict) -> dict[str, np.ndarray]:
    """Aligned arrays over the questions present in both; a missing fix answer falls back to greedy."""
    keys = sorted(greedy)
    arrays = {"dataset": np.array([k[0] for k in keys]), "greedy": np.array([greedy[k]["correct"] for k in keys])}
    for d in DETECTORS[:-1]:
        arrays[d] = np.array([greedy[k][d] for k in keys], dtype=float)
    arrays["fix"] = np.array([fix[k]["correct"] if k in fix else greedy[k]["correct"] for k in keys])
    arrays["fix_token"] = np.array([fix[k]["token_confidence"] if k in fix else -1.0 for k in keys])
    arrays["missing"] = int(sum(k not in fix for k in keys))
    return arrays


def final(arrays: dict, mask: np.ndarray, rule: str) -> np.ndarray:
    """Correctness of the final answer under one routing mask and rule."""
    take = mask if rule == "replace" else mask & (arrays["fix_token"] > arrays["token"])
    return np.where(take, arrays["fix"], arrays["greedy"])


def routed(arrays: dict, detector: str, budget: float, reference: dict, seed: int = 1) -> np.ndarray:
    """Questions sent to the fix: score at or below the validation threshold (random: probability budget)."""
    if detector == "random":
        return random_gate(len(arrays["greedy"]), budget, seed)
    return gate(arrays[detector], threshold(reference[detector], budget))


def outcome(arrays: dict, detector: str, rule: str, budget: float, reference: dict) -> np.ndarray:
    """Per-question correctness; for the random detector, one draw (used for paired tests)."""
    return final(arrays, routed(arrays, detector, budget, reference), rule)


def accuracy(arrays: dict, detector: str, rule: str, budget: float, reference: dict) -> float:
    """Pooled accuracy; the random detector is averaged over RANDOM_DRAWS draws."""
    if detector != "random":
        return float(outcome(arrays, detector, rule, budget, reference).mean())
    return float(np.mean([final(arrays, routed(arrays, "random", budget, reference, s), rule).mean() for s in range(RANDOM_DRAWS)]))


def curves(arrays: dict, reference: dict) -> dict:
    """Accuracy at every budget for every detector and rule, thresholds taken from `reference`."""
    return {d: {r: [accuracy(arrays, d, r, b, reference) for b in BUDGETS] for r in RULES} for d in DETECTORS}


def breakdown(arrays: dict, chosen: np.ndarray) -> dict:
    """The paired comparison against greedy, pooled and per dataset."""
    rows = {"pooled": policy_row(arrays["greedy"].tolist(), chosen.tolist())}
    for d in DATASET_NAMES:
        m = arrays["dataset"] == d
        if m.any():
            rows[d] = policy_row(arrays["greedy"][m].tolist(), chosen[m].tolist())
    return rows


def detector_quality(arrays: dict) -> dict:
    """AUROC of each detector for 'the 2B answer is right' and for 'the fix would help' (greedy wrong, fix right)."""
    helps = (~arrays["greedy"] & arrays["fix"]).astype(int)
    out = {}
    for d in DETECTORS[:-1]:
        out[d] = {"greedy_correct": auroc(arrays["greedy"].astype(int), arrays[d]),
                  "fix_helps": auroc(helps, -arrays[d]) if 0 < helps.sum() < len(helps) else None}
    return out


def headline(val: dict, test: dict, val_curves: dict, name: str, budget: float | None) -> dict:
    """One validation-chosen setting read once on test. budget None: the budget is chosen too."""
    options = [(d, r, b) for d in DETECTORS if d != "random" for r in RULES
               for b in (BUDGETS if budget is None else [budget])]
    d, r, b = max(options, key=lambda o: (val_curves[o[0]][o[1]][BUDGETS.index(o[2])], -o[2]))
    return read_test(val, test, val_curves, name, d, r, b)


def read_test(val: dict, test: dict, val_curves: dict, name: str, d: str, r: str, b: float) -> dict:
    """One fixed (detector, rule, budget) on test, with the validation threshold."""
    chosen = outcome(test, d, r, b, val)
    return {"name": name, "detector": d, "rule": r, "budget": b, "test_share": float(routed(test, d, b, val).mean()),
            "val_accuracy": val_curves[d][r][BUDGETS.index(b)], "test": breakdown(test, chosen),
            "test_random_same_budget": accuracy(test, "random", r, b, val),
            "vs_random": paired_difference(outcome(test, "random", r, b, val).tolist(), chosen.tolist())}


def efficient(val: dict, test: dict, val_curves: dict, best: dict, keep: float = 0.8) -> dict:
    """The smallest budget whose validation gain keeps `keep` of the best setting's gain, same detector and rule."""
    d, r = best["detector"], best["rule"]
    base = float(val["greedy"].mean())
    target = base + keep * (best["val_accuracy"] - base)
    b = next(b for b, a in zip(BUDGETS, val_curves[d][r]) if a >= target)
    return read_test(val, test, val_curves, f"smallest budget keeping {int(keep * 100)}% of the best validation gain", d, r, b)


def cost(detector: str, rule: str, share: float, fix: str) -> float:
    """Approximate compute in 2B greedy passes per question, for the share actually routed. A
    pre-generation detector that replaces never needs the 2B answer of a routed question."""
    skip = detector.startswith("pre_") and rule == "replace"
    return (1 - share if skip else 1.0) + share * FIX_COST.get(fix, 1.0)


def analyze(fix: str, greedy: dict, fixes: dict) -> dict:
    """Every headline of one fix."""
    val, test = table(greedy["val"], fixes["val"][fix]), table(greedy["test"], fixes["test"][fix])
    val_curves, test_curves = curves(val, val), curves(test, val)
    heads = [headline(val, test, val_curves, f"best detector and rule at a {int(b * 100)}% budget", b) for b in (0.2, 0.3)]
    best = headline(val, test, val_curves, "best detector, rule and budget", None)
    heads += [best, efficient(val, test, val_curves, best)]
    for h in heads:
        h["cost"] = cost(h["detector"], h["rule"], h["test_share"], fix)
    everything = final(test, np.ones(len(test["greedy"]), dtype=bool), "replace")
    oracle = test["greedy"] | test["fix"]
    return {"fix": fix, "questions": {"val": len(val["greedy"]), "test": len(test["greedy"])},
            "missing_answers": {"val": val["missing"], "test": test["missing"]},
            "greedy_test_accuracy": float(test["greedy"].mean()),
            "fix_everything": breakdown(test, everything), "oracle": breakdown(test, oracle),
            "detector_quality_test": detector_quality(test), "headlines": heads,
            "curves": {"budgets": BUDGETS, "val": val_curves, "test": test_curves}}


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.2f}"


def markdown(result: dict) -> str:
    """Headline tables for every fix."""
    lines = ["# Escalation study: our detector, new fixes", "",
             "Every setting is chosen on validation and read once on test. A budget sets a fixed threshold on the detector score "
             "from validation; on test each question is routed on its own score. Accuracy is pooled over the four datasets.", ""]
    for fix, r in result["fixes"].items():
        lines += [f"## Fix: {fix}", "",
                  f"Questions {r['questions']}, missing fix answers {r['missing_answers']}. 2B greedy on test: {pct(r['greedy_test_accuracy'])}. "
                  f"Fix everything: {pct(r['fix_everything']['pooled']['accuracy'])} ({pts(r['fix_everything']['pooled']['difference'])}). "
                  f"Oracle (either answer right): {pct(r['oracle']['pooled']['accuracy'])}.", "",
                  "| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget |",
                  "|---|---|---|---:|---:|---:|---|---|---|"]
        for h in r["headlines"]:
            t = h["test"]["pooled"]
            v = h["vs_random"]
            lines.append(f"| {h['name']} | {h['detector']} | {h['rule']} | {int(round(100 * h['budget']))}% / {100 * h['test_share']:.1f}% | {h['cost']:.2f} | {pct(t['accuracy'])} | "
                         f"{pts(t['difference'])} ({pts(t['ci_low'])} to {pts(t['ci_high'])}) | +{t['fixed']} / -{t['broken']} | "
                         f"{pts(v['difference'])} ({pts(v['ci_low'])} to {pts(v['ci_high'])}) |")
        lines += ["", "Per dataset, first headline (20% budget):", "", "| Dataset | Greedy | Chosen | Gain (95% interval) | Fix everything |", "|---|---:|---:|---|---:|"]
        h = r["headlines"][0]
        for d in DATASET_NAMES:
            if d in h["test"]:
                t = h["test"][d]
                lines.append(f"| {d} | {pct(t['accuracy'] - t['difference'])} | {pct(t['accuracy'])} | {pts(t['difference'])} ({pts(t['ci_low'])} to {pts(t['ci_high'])}) | "
                             f"{pct(r['fix_everything'][d]['accuracy'])} |")
        lines += ["", "Detector AUROC on test (2B answer right / fix would help):", "", "| Detector | 2B right | Fix helps |", "|---|---:|---:|"]
        for d, q in r["detector_quality_test"].items():
            helps = "n/a" if q["fix_helps"] is None else f"{q['fix_helps']:.3f}"
            lines.append(f"| {d} | {q['greedy_correct']:.3f} | {helps} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Load greedy answers and fixes, analyze each fix, write JSON and Markdown."""
    parser = argparse.ArgumentParser(description="Routing study: send the least-confident questions to a fix.")
    parser.add_argument("--output", type=Path, default=RESULTS / "escalation.json")
    args = parser.parse_args()
    greedy = {s: load_greedy(s) for s in ("val", "test")}
    fixes = {s: load_fixes(s) for s in ("val", "test")}
    names = sorted(set(fixes["val"]) & set(fixes["test"]))
    print({s: len(g) for s, g in greedy.items()}, names, flush=True)
    result = {"budgets": BUDGETS, "fix_cost": FIX_COST, "fixes": {f: analyze(f, greedy, fixes) for f in names}}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"[done] wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
