# export_heads.py
#
# Retrains the winning head configurations on the pooled mixture and saves their weights,
# so a trained correctness head can be loaded without rerunning the sweeps. The sweeps keep
# weights only in memory, so this reads each winner's configuration from the sweep results
# and trains it again (seed 1, same data and early stopping). It then checks that the test
# AUROC of the saved head matches the AUROC reported for that winner.
# Saved per head: the network weights, the head spec, the temperature of each mode, and the
# reproduced and reported test AUROC.
#
# Heads written to trained_heads/approach1/<kind>_<old|new_swap|new_noise>.pt:
#   old        old loss (no grounding term), the same head for both corruptions
#   new_swap   new loss with the swapped-image grounding term
#   new_noise  new loss with the noised-image grounding term
# Usage: python -m experiments.approach1_latent_self_correction.export_heads
# load_head() below returns a ready head and its temperatures from a saved file.

import argparse
import json
from pathlib import Path

import torch

from experiments.approach1_latent_self_correction.mixture_heads import (
    HeadSpec,
    build_head,
    head_logits,
    modes_for,
    train_head,
)
from experiments.approach1_latent_self_correction.sweep_mixture import FEATURE_DIR, MODELS_DIR, load_split
from experiments.approach1_latent_self_correction.unified_q_head import to_probabilities

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "trained_heads" / "approach1"
KINDS = ("pre", "verifier", "unified")
MODE = {"pre": "main", "verifier": "main", "unified": "joint"}
# output label -> (sweep tag, winner group)
EXPORTS = {"old": ("mixture", "old"), "new_swap": ("mixture", "new"), "new_noise": ("mixture_noise", "new")}
SEED = 1


def load_head(path: Path, device: str = "cpu"):
    """Load a saved head and return the network, its spec, and the temperature of each mode."""
    saved = torch.load(path, map_location=device, weights_only=False)
    spec = HeadSpec(**saved["spec"])
    head = build_head(spec, saved["feature_size"]).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()
    return head, spec, saved["temperatures"]


def score(head, spec: HeadSpec, temperatures: dict, prompt: torch.Tensor, generated: torch.Tensor | None = None) -> dict:
    """Probabilities of being correct per mode: the prompt state, and for the verifier and joint modes the answer state."""
    if spec.kind == "verifier" and generated is None:
        raise ValueError("the verifier head needs the answer state")
    modes = modes_for(spec.kind) if generated is not None or spec.kind != "unified" else ("pre_generation",)
    with torch.no_grad():
        logits = head_logits(head, spec.kind, prompt.float(), None if generated is None else generated.float(), modes)
    return {mode: to_probabilities(logit.float().cpu().numpy(), temperatures[mode]) for mode, logit in logits.items()}


def export_one(kind: str, label: str, tag: str, group: str, data: dict, device: torch.device, output: Path) -> dict:
    """Train one winner again, save it, and report how closely its test AUROC matches the sweep."""
    sweep = json.loads((MODELS_DIR / f"{tag}_{kind}_results.json").read_text(encoding="utf-8"))
    winner = sweep["winners"][group]
    spec = HeadSpec(**{**winner["config"], "seed": SEED})
    run = train_head(spec, data["train"], data["validation"], data["test"], data["names"], device, keep_state=True)
    mode = MODE[kind]
    reproduced = run["test"]["pooled"][mode]["correctness_auroc"]
    reported = winner["test"][mode]["correctness_auroc"]["mean"]
    saved = {
        "kind": kind,
        "label": label,
        "spec": run["spec"],
        "feature_size": int(data["train"]["prompt"].shape[1]),
        "state_dict": run["state"],
        "temperatures": run["temperatures"],
        "epoch": run["epoch"],
        "test_auroc_reproduced": reproduced,
        "test_auroc_reported_mean": reported,
        "datasets": data["names"],
    }
    path = output / f"{kind}_{label}.pt"
    torch.save(saved, path)
    print(f"[saved] {path.name}: test AUROC {reproduced:.4f} (sweep mean over three seeds {reported:.4f})", flush=True)
    return {"file": path.name, "kind": kind, "label": label, "config": winner["config"], "test_auroc": reproduced, "sweep_mean_auroc": reported}


def main() -> None:
    """Export every winner listed in EXPORTS for every head kind."""
    parser = argparse.ArgumentParser(description="Save trained heads for the pooled-mixture winners.")
    parser.add_argument("--size", default="2b")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = []
    loaded = {}
    for label, (tag, group) in EXPORTS.items():
        if tag not in loaded:
            train, names = load_split(FEATURE_DIR / f"{tag}_train_{args.size}.pt", device)
            validation, _ = load_split(FEATURE_DIR / f"{tag}_val_{args.size}.pt", device)
            test, _ = load_split(FEATURE_DIR / f"{tag}_test_{args.size}.pt", device)
            loaded[tag] = {"train": train, "validation": validation, "test": test, "names": names}
        for kind in KINDS:
            manifest.append(export_one(kind, label, tag, group, loaded[tag], device, args.output))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] {len(manifest)} heads in {args.output}")


if __name__ == "__main__":
    main()
