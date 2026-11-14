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
# Cost is counted per question in model passes, as in report 11 and from what each question actually
# needed: 1 greedy generation, plus the detector's pass if it needs one; a corrected question adds one
# batched sample call (k > 0), one call per correction (contrastive decoding 2, grounding 1 for the box
# plus 1 per grounded answer, reasoning 1) and two scoring passes per distinct candidate (encoding and
# judge). Generation calls are also reported on their own, since a scoring pass is a single forward pass.
#
# Configurations: the validation choice of report 12 (T1_k5_everything), the cost-limited choice
# (T1_k0_ground_cot), reasoning only (T1_k0_cot) and crops + reasoning (T1_k0_zoom_cot), all with the
# v3 scalars_ds ranker. Headlines, each chosen on validation and read once on test:
#   per configuration   the best detector at budgets 20, 30 and 50%; the smallest budget whose
#                       validation gain keeps 90% of correcting everything
#   overall             the best (configuration, detector, budget) under an average cost of 2, 3 and 5
#                       passes per question
# Writes results/gating_v3.json and results/gating_v3.md.

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
from experiments.approach1_latent_self_correction.train_ranker_v3 import SETS, RankerV3, candidate_set, choose

CONFIGS = ("T1_k5_everything", "T1_k0_ground_cot", "T1_k0_cot", "T1_k0_zoom_cot")
DETECTORS = ("token", "pre_head", "after_head", "judge", "random")
DETECTOR_PASS = {"token": 0, "pre_head": 0, "after_head": 1, "judge": 1, "random": 0}
FIXED_BUDGETS = (0.2, 0.3, 0.5)
COST_CAPS = (2.0, 3.0, 5.0)
KEEP = 0.9
RANDOM_DRAWS = 50
VARIANT = "scalars_ds"


def parse(key: str) -> tuple[str, int, str]:
    """('1', 5, 'everything') from 'T1_k5_everything'."""
    temperature, rest = key[1:].split("_k", 1)
    k, name = rest.split("_", 1)
    return temperature, int(k), name


def generation_calls(k: int, sources: tuple) -> int:
    """Generation calls a corrected question adds to its greedy answer."""
    calls = 1 if k else 0
    calls += sum(2 if s.startswith("vcd") else 1 for s in sources)
    calls += 1 if any(s.startswith("ground") for s in sources) else 0  # the bounding-box request
    return calls


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


def load_models(device) -> list[RankerV3]:
    """The three seeds of the v3 scalars_ds ranker saved by train_ranker_v3.py."""
    saved = torch.load(RESULTS / "candidate_ranker_v3_models.pt", map_location=device, weights_only=False)[VARIANT]
    models = []
    for state in saved:
        model = RankerV3(VARIANT, state["network.0.weight"].shape[0]).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def split_table(questions: list[dict], models: list, device) -> dict:
    """Per question: dataset, greedy and corrected correctness per configuration, cost, detector scores."""
    table = {"dataset": np.array([q["key"][0] for q in questions]),
             "greedy": np.array([bool(q["encoded"][q["greedy"]]["correct"]) for q in questions]),
             "scores": {d: np.array([q["gates"][d] for q in questions], dtype=float) for d in DETECTORS if d != "random"},
             "corrected": {}, "passes": {}, "calls": {}}
    for key in CONFIGS:
        temperature, k, name = parse(key)
        config = (temperature, k, SETS[name])
        table["corrected"][key] = np.array(choose(models, questions, config, device))
        candidates = np.array([len(candidate_set(q, temperature, k, SETS[name])[0]) for q in questions])
        table["calls"][key] = 1 + generation_calls(k, SETS[name])
        table["passes"][key] = table["calls"][key] + 2 * candidates
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
    return {"final": final, "share": float(mask.mean()), "passes": float(passes.mean()), "calls": float(calls.mean())}


def summary(table: dict, key: str, detector: str, budget: float, reference: dict) -> dict:
    """Accuracy, share and costs; the random detector averaged over draws."""
    if detector != "random":
        e = evaluate(table, key, detector, budget, reference)
        return {"accuracy": float(e["final"].mean()), "share": e["share"], "passes": e["passes"], "calls": e["calls"]}
    draws = [evaluate(table, key, "random", budget, reference, s) for s in range(RANDOM_DRAWS)]
    return {m: float(np.mean([d["final"].mean() if m == "accuracy" else d[m] for d in draws])) for m in ("accuracy", "share", "passes", "calls")}


def grid(table: dict, reference: dict) -> dict:
    """Every configuration, detector and budget."""
    return {key: {d: [summary(table, key, d, b, reference) for b in BUDGETS] for d in DETECTORS} for key in CONFIGS}


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
            "test_share": e["share"], "test_passes": e["passes"], "test_calls": e["calls"],
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
    for key in CONFIGS:
        out.append(read_test(val, test, val_grid, "correct every question (no gate)", key, "token", 1.0))
        for b in FIXED_BUDGETS:
            out.append(read_test(val, test, val_grid, f"best detector at a {int(b * 100)}% budget", key, best_detector(val_grid, key, b), b))
        full = val_grid[key]["token"][-1]["accuracy"]
        target = base + KEEP * (full - base)
        b = next(b for b in BUDGETS if val_grid[key][best_detector(val_grid, key, b)][BUDGETS.index(b)]["accuracy"] >= target)
        out.append(read_test(val, test, val_grid, f"smallest budget keeping {int(KEEP * 100)}% of the full validation gain", key,
                             best_detector(val_grid, key, b), b))
    for cap in COST_CAPS:
        options = [(key, d, b) for key in CONFIGS for d in DETECTORS if d != "random" for b in BUDGETS
                   if val_grid[key][d][BUDGETS.index(b)]["passes"] <= cap]
        key, d, b = max(options, key=lambda o: (val_grid[o[0]][o[1]][BUDGETS.index(o[2])]["accuracy"], -o[2]))
        out.append(read_test(val, test, val_grid, f"best under {cap:g} passes per question (any configuration)", key, d, b))
    return out


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.2f}"


def markdown(result: dict) -> str:
    """Headline table and per-dataset rows of the gated settings."""
    lines = ["# Gating the v3 correction step", "",
             "Thresholds on the detector score come from validation; on test each question is gated on its own score. "
             f"Ranker: v3 {VARIANT}. Greedy on test: {pct(result['greedy_test'])}. Passes: model passes per question "
             "(generation calls plus two scoring passes per distinct candidate of a corrected question); calls: generation calls only.", "",
             "| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |",
             "|---|---|---|---|---:|---:|---:|---|---|---|"]
    for h in result["headlines"]:
        t, v = h["test"]["pooled"], h["vs_random"]
        lines.append(f"| {h['name']} | {h['configuration']} | {h['detector']} | {int(round(100 * h['budget']))}% / {100 * h['test_share']:.1f}% | "
                     f"{h['test_passes']:.2f} | {h['test_calls']:.2f} | {pct(t['accuracy'])} | {pts(t['difference'])} ({pts(t['ci_low'])} to {pts(t['ci_high'])}) | "
                     f"+{t['fixed']} / -{t['broken']} | {pts(v['difference'])} ({pts(v['ci_low'])} to {pts(v['ci_high'])}) |")
    lines += ["", "Per dataset (gain over greedy on test):", "", "| Setting | Configuration | " + " | ".join(DATASET_NAMES) + " |",
              "|---|---|" + "---|" * len(DATASET_NAMES)]
    for h in result["headlines"]:
        lines.append(f"| {h['name']} | {h['configuration']} | " + " | ".join(
            f"{pts(h['test'][d]['difference'])} ({pts(h['test'][d]['ci_low'])} to {pts(h['test'][d]['ci_high'])})" for d in DATASET_NAMES) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Score every configuration once per split, then gate."""
    parser = argparse.ArgumentParser(description="Gating for the v3 correction step.")
    parser.add_argument("--output", type=Path, default=RESULTS / "gating_v3.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    tables = {}
    for split in ("val", "test"):
        questions = merge_v2(load_questions(split), split)
        attach_gates(questions, split)
        tables[split] = split_table(questions, models, device)
        print(f"[scored] {split}: {len(questions)} questions", flush=True)
    val, test = tables["val"], tables["test"]
    val_grid, test_grid = grid(val, val), grid(test, val)
    result = {"budgets": list(BUDGETS), "configurations": list(CONFIGS), "detectors": list(DETECTORS),
              "greedy_val": float(val["greedy"].mean()), "greedy_test": float(test["greedy"].mean()),
              "ungated_test_accuracy": {k: float(test["corrected"][k].mean()) for k in CONFIGS},
              "headlines": headlines(val, test, val_grid), "curves": {"val": val_grid, "test": test_grid}}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"[done] wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
