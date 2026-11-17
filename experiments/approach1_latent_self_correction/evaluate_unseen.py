# evaluate_unseen.py
#
# Transfer test: the whole correction pipeline, frozen, on datasets it never trained on (AI2D and
# InfographicVQA, split "unseen"; build_unseen_splits.py). Nothing is trained, tuned or chosen on these
# datasets:
#   ranker        the saved v3 rankers, *_nods variant (it never sees a dataset label, so an unseen
#                 dataset is not an unknown input); sampled reasoning (candidate_ranker_v3_multi_models.pt)
#                 and, when present, the 8B and upscaled candidates (candidate_ranker_v3_multi_esc_models.pt)
#                 and the 8B reasoning answers without the judge input (candidate_ranker_v3_multi_esc_cot8b_drop_judge_models.pt,
#                 used once results/correction_v5/unseen exists)
#   candidates    the configuration that ranker's own run chose on in-domain validation, plus a fixed list
#   gate          answer-probability thresholds taken from the IN-DOMAIN validation split (gating.py), at
#                 30% (the budget the in-domain gating study chose) and 50%
#   baselines     the training-free baselines of compute_baselines.py and the 8B model, on the same questions
# Every row is compared with the 2B greedy answer on the same questions (paired bootstrap, McNemar).
#
# Writes results/unseen.json and results/unseen.md.

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction import train_ranker_v3 as v3
from experiments.approach1_latent_self_correction.compute_baselines import evaluate as baseline_rows
from experiments.approach1_latent_self_correction.compute_baselines import larger_models
from experiments.approach1_latent_self_correction.correction_scale import UNSEEN_DATASETS
from experiments.approach1_latent_self_correction.correction_stats import policy_row
from experiments.approach1_latent_self_correction.gate_ranker_v3 import load_models, parse
from experiments.approach1_latent_self_correction.gating import combine, gate, threshold
from experiments.approach1_latent_self_correction.train_candidate_ranker import load_questions
from experiments.approach1_latent_self_correction.train_ranker_v2 import RESULTS, merge_v2

SPLIT = "unseen"
VARIANT = "scalars_nods"
BUDGETS = (0.3, 0.5)
FIXED = {"multi": ("T1_k0_cot_multi", "T1_k0_ground_cot_multi", "T1_k0_everything_multi"),
         "multi_esc": ("T1_k0_esc8b", "T1_k0_ground_cot_multi_esc8b", "T1_k0_ground_cot_multi_esc", "T1_k0_everything_multi_esc"),
         "cot8b_nojudge": ("T1_k0_cot8b", "T1_k0_ground_cot_multi_cot8b", "T1_k0_ground_cot_multi_esc8b_cot8b")}
HEADLINE = "candidate_ranker_v3_multi_esc_cot8b_drop_judge"  # the best in-domain ranker: 8B answer + 8B reasoning, no judge


def in_domain_choice(report: Path) -> str:
    """The configuration with the best in-domain validation accuracy for the nods variant (never an unseen number)."""
    result = json.loads(report.read_text())
    return max(result["val"], key=lambda key: result["val"][key].get(f"ranker_v3_{VARIANT}", {"pooled": {"accuracy": -1}})["pooled"]["accuracy"])


def val_token_scores() -> np.ndarray:
    """Answer probability of the greedy answer on every in-domain validation question (the gate's reference)."""
    scores = []
    for path in sorted((RESULTS / "pools" / "T1" / "val").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line) if line.strip() else {}
            if "greedy" in record:
                scores.append(record["greedy"]["token_confidence"])
    return np.array(scores)


def token_scores(questions: list[dict]) -> np.ndarray:
    """Answer probability of the greedy answer of each unseen question, from its T=1.0 pool."""
    pools = {}
    for path in sorted((RESULTS / "pools" / "T1" / SPLIT).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line) if line.strip() else {}
            if "greedy" in record:
                pools[(record["dataset"], record["index"])] = record["greedy"]["token_confidence"]
    return np.array([pools[q["key"]] for q in questions])


def breakdown(datasets: np.ndarray, greedy: np.ndarray, final: np.ndarray) -> dict:
    """Paired comparison with greedy, pooled and per unseen dataset."""
    out = {"pooled": policy_row(greedy.tolist(), final.tolist())}
    for d in UNSEEN_DATASETS:
        m = datasets == d
        if m.any():
            out[d] = policy_row(greedy[m].tolist(), final[m].tolist())
    return out


def score_configs(questions: list[dict], models: list, keys: list[str], device) -> dict[str, np.ndarray]:
    """Per-question correctness of the ranker's top candidate for each configuration."""
    out = {}
    for key in keys:
        temperature, k, name = parse(key)
        out[key] = np.array(v3.choose(models, questions, (temperature, k, v3.SETS[name]), device))
    return out


def main() -> None:
    """Score the frozen pipeline and the baselines on the unseen datasets."""
    parser = argparse.ArgumentParser(description="Transfer test of the frozen correction pipeline on unseen datasets.")
    parser.add_argument("--output", type=Path, default=RESULTS / "unseen.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3.SETS.update(v3.MULTI_SETS)

    questions = merge_v2(load_questions(SPLIT), SPLIT)
    v3.merge_cot_samples(questions, SPLIT)
    escalation = (RESULTS / "candidate_ranker_v3_multi_esc_models.pt").exists() and (v3.V4_DIR / SPLIT).exists()
    if escalation:
        v3.merge_cot_samples(questions, SPLIT, v3.V4_DIR)
    datasets = np.array([q["key"][0] for q in questions])
    greedy = np.array([bool(q["encoded"][q["greedy"]]["correct"]) for q in questions])
    print({d: int((datasets == d).sum()) for d in UNSEEN_DATASETS}, "questions", flush=True)

    # Phase 1: rankers trained without the escalation candidates (their feature width has no escalation column).
    rows, chosen = {}, {}
    chosen["multi"] = in_domain_choice(RESULTS / "candidate_ranker_v3_multi.json")
    keys = list(dict.fromkeys((chosen["multi"],) + FIXED["multi"]))
    outcomes = score_configs(questions, load_models(RESULTS / "candidate_ranker_v3_multi_models.pt", VARIANT, device), keys, device)
    # Phase 2: the rankers trained with the escalation candidates; enabling them widens the features.
    if escalation:
        v3.enable_escalation()
        chosen["multi_esc"] = in_domain_choice(RESULTS / "candidate_ranker_v3_multi_esc.json")
        keys = list(dict.fromkeys((chosen["multi_esc"],) + FIXED["multi_esc"]))
        esc_models = load_models(RESULTS / "candidate_ranker_v3_multi_esc_models.pt", VARIANT, device)
        outcomes.update({f"{k} (with 8B)" if k in outcomes else k: v for k, v in score_configs(questions, esc_models, keys, device).items()})
    # Phase 3: the ranker with the 8B reasoning answers, trained without the judge; merged last, so the earlier
    # phases see exactly the candidates they were trained with.
    if escalation and (RESULTS / f"{HEADLINE}_models.pt").exists() and (v3.V5_DIR / SPLIT).exists():
        v3.merge_cot_samples(questions, SPLIT, v3.V5_DIR)
        v3.DROPPED.update(v3.FEATURE_GROUPS["judge"])
        chosen["cot8b_nojudge"] = in_domain_choice(RESULTS / f"{HEADLINE}.json")
        keys = list(dict.fromkeys((chosen["cot8b_nojudge"],) + FIXED["cot8b_nojudge"]))
        headline_models = load_models(RESULTS / f"{HEADLINE}_models.pt", VARIANT, device)
        outcomes.update({f"{k} (8B reasoning, no judge)": v for k, v in score_configs(questions, headline_models, keys, device).items()})

    reference, tokens = val_token_scores(), token_scores(questions)
    for key, corrected in outcomes.items():
        rows[f"{key}, every question"] = breakdown(datasets, greedy, corrected)
        for b in BUDGETS:
            routed = gate(tokens, threshold(reference, b))
            rows[f"{key}, gate {int(b * 100)}% (corrected {100 * routed.mean():.1f}%)"] = breakdown(datasets, greedy, combine(routed, corrected, greedy))

    larger = larger_models(SPLIT)
    if "8b" in larger:
        big = np.array([bool(larger["8b"][q["key"]]["correct"]) if q["key"] in larger["8b"] else g for q, g in zip(questions, greedy)])
        for b in BUDGETS:
            routed = gate(tokens, threshold(reference, b))
            rows[f"8B on gated questions, gate {int(b * 100)}% (corrected {100 * routed.mean():.1f}%)"] = breakdown(datasets, greedy, combine(routed, big, greedy))
    baselines = baseline_rows(questions, larger)
    result = {"questions": {d: int((datasets == d).sum()) for d in UNSEEN_DATASETS}, "variant": VARIANT,
              "in_domain_choice": chosen, "pipeline": rows,
              "baselines": {name: {k: v for k, v in e.items() if k in ("calls", "passes", "pooled")} | {d: e.get(d) for d in UNSEEN_DATASETS if d in e}
                            for name, e in baselines.items()}}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"[done] wrote {args.output} and {args.output.with_suffix('.md')}")


def markdown(result: dict) -> str:
    """Pipeline and baseline tables, pooled and per unseen dataset."""
    pct = lambda x: f"{100 * x:.2f}%"
    ci = lambda r: f"{100 * r['difference']:+.2f} ({100 * r['ci_low']:+.2f} to {100 * r['ci_high']:+.2f})"
    names = [d for d in UNSEEN_DATASETS if result["questions"].get(d)]
    lines = ["# Transfer to unseen datasets (frozen pipeline)", "",
             f"Questions: {result['questions']}. Ranker variant: {result['variant']} (no dataset input). Configurations chosen on in-domain "
             f"validation: {result['in_domain_choice']}. Gate thresholds from in-domain validation.", "",
             "| Setting | Accuracy | Gain over greedy (95% interval) | Fixed / broken | " + " | ".join(names) + " |", "|---|---:|---|---|" + "---|" * len(names)]
    for name, r in list(result["pipeline"].items()) + [(f"baseline: {n}", e) for n, e in result["baselines"].items()]:
        p = r["pooled"]
        lines.append(f"| {name} | {pct(p['accuracy'])} | {ci(p)} | +{p['fixed']} / -{p['broken']} | " + " | ".join(ci(r[d]) for d in names) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
