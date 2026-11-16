# analyze_cascade.py
#
# A two-step escalation cascade, 2B -> 4B -> 8B, against sending questions straight to 8B
# (analyze_escalation.py). It needs no new generation: the 2B greedy answers with their detector scores
# (results/pools/T1) and the 4B and 8B greedy answers of escalation.py on every validation and test question.
#
#   step 1   the 2B detector flags the b1 least-confident questions (validation threshold, gating.py);
#            those get a 4B answer
#   step 2   of the flagged questions, the b2 share whose 4B answer has the lowest token confidence
#            (threshold again from validation, on the flagged validation questions) also get an 8B answer
#   answer   rule "replace": the answer of the last model asked; rule "confident": the answer with the
#            highest token confidence among the models asked
#   cost     in 2B passes: 1 + 2 x share sent to 4B + 4 x share sent to 8B (FIX_COST of analyze_escalation.py)
#
# The direct route (2B -> 8B at budget b, cost 1 + 4b) is computed with the same code (b2 = 1 and no 4B
# step), so both are on one accuracy-against-cost axis. Every setting (detector, rule, b1, b2) is chosen
# on validation; test is read once per headline: the best validation accuracy under an average cost of
# 1.5, 2, 2.5, 3 and 4 passes, for the cascade and for the direct route, with the paired difference
# between the two on test. Writes results/cascade.json and results/cascade.md.

import itertools
import json

import numpy as np

from experiments.approach1_latent_self_correction.analyze_escalation import (
    DETECTORS,
    FIX_COST,
    RESULTS,
    load_fixes,
    load_greedy,
)
from experiments.approach1_latent_self_correction.correction_scale import DATASET_NAMES
from experiments.approach1_latent_self_correction.correction_stats import paired_difference, policy_row
from experiments.approach1_latent_self_correction.gating import BUDGETS, gate, threshold

SCORED = tuple(d for d in DETECTORS if d != "random")
RULES = ("replace", "confident")
COST_CAPS = (1.5, 2.0, 2.5, 3.0, 4.0)


def arrays(split: str) -> dict:
    """Aligned per-question arrays: 2B correctness, detectors and token confidence; 4B and 8B correctness
    and token confidence. Only questions that have both larger answers are kept (all of them, in practice)."""
    greedy, fixes = load_greedy(split), load_fixes(split)
    keys = sorted(k for k in greedy if k in fixes["4b"] and k in fixes["8b"])
    out = {"dataset": np.array([k[0] for k in keys]), "missing": len(greedy) - len(keys)}
    out["c2"] = np.array([greedy[k]["correct"] for k in keys])
    for d in SCORED:
        out[d] = np.array([greedy[k][d] for k in keys], dtype=float)
    for size in ("4b", "8b"):
        out[f"c{size[0]}"] = np.array([bool(fixes[size][k]["correct"]) for k in keys])
        out[f"t{size[0]}"] = np.array([fixes[size][k]["token_confidence"] for k in keys], dtype=float)
    return out


def masks(a: dict, ref: dict, detector: str, b1: float, b2: float, direct: bool) -> tuple[np.ndarray, np.ndarray]:
    """(sent to 4B, sent to 8B). Thresholds come from `ref` (validation): b1 on the 2B detector, b2 on the 4B
    token confidence of the validation questions that step 1 flags. The direct route skips 4B."""
    step1 = gate(a[detector], threshold(ref[detector], b1))
    if direct:
        return np.zeros_like(step1), step1
    flagged = ref["t4"][gate(ref[detector], threshold(ref[detector], b1))]
    if b2 <= 0 or flagged.size == 0:
        return step1, np.zeros_like(step1)
    return step1, step1 & gate(a["t4"], threshold(flagged, b2))


def final(a: dict, to4: np.ndarray, to8: np.ndarray, rule: str) -> np.ndarray:
    """Correctness of the answer each question ends with."""
    if rule == "replace":
        return np.where(to8, a["c8"], np.where(to4, a["c4"], a["c2"]))
    t2 = a["token"]
    t4 = np.where(to4, a["t4"], -1.0)
    t8 = np.where(to8, a["t8"], -1.0)
    best = np.argmax(np.stack([t2, t4, t8]), axis=0)  # ties go to the smaller model
    return np.choose(best, [a["c2"], a["c4"], a["c8"]])


def setting(a: dict, ref: dict, detector: str, rule: str, b1: float, b2: float, direct: bool) -> dict:
    """Accuracy, cost and shares of one setting, with per-question correctness for the paired tests."""
    to4, to8 = masks(a, ref, detector, b1, b2, direct)
    correct = final(a, to4, to8, rule)
    cost = 1.0 + FIX_COST["4b"] * to4.mean() + FIX_COST["8b"] * to8.mean()
    return {"correct": correct, "accuracy": float(correct.mean()), "cost": float(cost),
            "to4": float(to4.mean()), "to8": float(to8.mean())}


def grid(val: dict) -> list[dict]:
    """Every cascade and direct setting on validation (thresholds from validation)."""
    rows = []
    for detector, rule, b1 in itertools.product(SCORED, RULES, BUDGETS):
        if b1 == 0:
            continue
        s = setting(val, val, detector, rule, b1, 1.0, True)
        rows.append({"route": "direct", "detector": detector, "rule": rule, "b1": b1, "b2": None, "accuracy": s["accuracy"], "cost": s["cost"]})
        for b2 in BUDGETS:
            s = setting(val, val, detector, rule, b1, b2, False)
            rows.append({"route": "cascade", "detector": detector, "rule": rule, "b1": b1, "b2": b2, "accuracy": s["accuracy"], "cost": s["cost"]})
    return rows


def breakdown(a: dict, correct: np.ndarray) -> dict:
    """Paired comparison with the 2B greedy answer, pooled and per dataset."""
    out = {"pooled": policy_row(a["c2"].tolist(), correct.tolist())}
    for d in DATASET_NAMES:
        m = a["dataset"] == d
        if m.any():
            out[d] = policy_row(a["c2"][m].tolist(), correct[m].tolist())
    return out


def read_test(val: dict, test: dict, row: dict) -> dict:
    """One validation-chosen setting on test."""
    s = setting(test, val, row["detector"], row["rule"], row["b1"], row["b2"] if row["b2"] is not None else 1.0, row["route"] == "direct")
    return {**row, "val_accuracy": row["accuracy"], "val_cost": row["cost"], "test": breakdown(test, s["correct"]),
            "test_cost": s["cost"], "test_to4": s["to4"], "test_to8": s["to8"], "correct": s["correct"]}


def headlines(val: dict, test: dict, rows: list[dict]) -> list[dict]:
    """Best validation setting of each route under each cost cap, read on test, with cascade minus direct."""
    out = []
    for cap in COST_CAPS:
        picked = {}
        for route in ("cascade", "direct"):
            fits = [r for r in rows if r["route"] == route and r["cost"] <= cap]
            if fits:
                picked[route] = read_test(val, test, max(fits, key=lambda r: (r["accuracy"], -r["cost"])))
        entry = {"cap": cap, **{k: {x: y for x, y in v.items() if x != "correct"} for k, v in picked.items()}}
        if len(picked) == 2:
            entry["cascade_minus_direct"] = paired_difference(picked["direct"]["correct"].tolist(), picked["cascade"]["correct"].tolist())
        out.append(entry)
    return out


def markdown(result: dict) -> str:
    """Headline table."""
    pct = lambda x: f"{100 * x:.2f}%"
    pts = lambda x: f"{100 * x:+.2f}"
    lines = ["# Escalation cascade: 2B -> 4B -> 8B against 2B -> 8B", "",
             f"Greedy 2B on test: {pct(result['greedy_test'])}; 4B everywhere {pct(result['all_4b_test'])}; 8B everywhere {pct(result['all_8b_test'])}. "
             "Cost in 2B passes (4B = 2, 8B = 4). Each setting is chosen on validation under the cost cap and read once on test.", "",
             "| Cost cap | Route | Detector | Rule | To 4B / to 8B (test) | Cost (test) | Test accuracy | Gain (95% interval) | Cascade minus direct |",
             "|---:|---|---|---|---|---:|---:|---|---|"]
    for h in result["headlines"]:
        for route in ("cascade", "direct"):
            if route not in h:
                continue
            r = h[route]
            p = r["test"]["pooled"]
            diff = h.get("cascade_minus_direct") if route == "cascade" else None
            d = f"{pts(diff['difference'])} ({pts(diff['ci_low'])} to {pts(diff['ci_high'])})" if diff else ""
            lines.append(f"| {h['cap']:g} | {route} | {r['detector']} | {r['rule']} | {100 * r['test_to4']:.1f}% / {100 * r['test_to8']:.1f}% | "
                         f"{r['test_cost']:.2f} | {pct(p['accuracy'])} | {pts(p['difference'])} ({pts(p['ci_low'])} to {pts(p['ci_high'])}) | {d} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Grid on validation, headlines on test."""
    val, test = arrays("val"), arrays("test")
    print(f"[loaded] val {len(val['c2'])} (missing {val['missing']}), test {len(test['c2'])} (missing {test['missing']})", flush=True)
    rows = grid(val)
    result = {"greedy_test": float(test["c2"].mean()), "all_4b_test": float(test["c4"].mean()), "all_8b_test": float(test["c8"].mean()),
              "missing": {"val": val["missing"], "test": test["missing"]}, "cost_caps": list(COST_CAPS),
              "headlines": headlines(val, test, rows), "validation_grid": rows}
    output = RESULTS / "cascade.json"
    output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))
    print(f"[done] wrote {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
