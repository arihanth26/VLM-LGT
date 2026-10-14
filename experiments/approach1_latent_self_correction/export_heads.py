# export_heads.py
#
# Retrains the trained correctness heads of the Approach 1 experiments and saves their
# weights, so any of them can be loaded without rerunning the sweeps (which keep weights
# only in memory). Two modes:
#   winners     for one tag (the pooled mixture, the noise mixture, or one dataset alone) and
#               every head kind: each winning configuration of the sweep (old, new, linear
#               probes, ablation cells), on all three seeds. Configurations shared by several
#               winner groups are trained once and listed under every group name.
#   sequential  for one tag and head kind: the head after the last stage of sequential training
#               (seed 1), for the old and new loss, with and without replay, on all three orders.
# Every winners head is checked: the test AUROC averaged over its three seeds must match the
# three-seed mean in the sweep results, and the manifest records both.
# Files: trained_heads/approach1/winners/<tag>/<kind>_<group>_seed<N>.pt and manifest.json,
#        trained_heads/approach1/sequential/<tag>/<kind>_<variant>_<order>_seed1.pt and manifest.
# Usage: python -m experiments.approach1_latent_self_correction.export_heads --mode winners --tag mixture
# load_head() and score() below load a saved head and score cached states with it.

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.approach1_latent_self_correction.continual import run_sequence, split_by_dataset
from experiments.approach1_latent_self_correction.mixture_heads import (
    HeadSpec,
    build_head,
    head_logits,
    modes_for,
    train_head,
)
from experiments.approach1_latent_self_correction.sweep_continual import ORDER_NAMES, load_config
from experiments.approach1_latent_self_correction.sweep_mixture import FEATURE_DIR, MODELS_DIR, SEEDS, load_split
from experiments.approach1_latent_self_correction.unified_q_head import to_probabilities

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "trained_heads" / "approach1"
KINDS = ("pre", "verifier", "unified")
MODE = {"pre": "main", "verifier": "main", "unified": "joint"}
GROUP_ORDER = ("old", "new", "old_linear", "new_linear", "new_full", "cell_plain", "cell_contrast", "cell_ground", "cell_full")
VARIANTS = {"old_replay0": ("old", 0.0), "old_replay0.1": ("old", 0.1), "new_replay0": ("new", 0.0), "new_replay0.1": ("new", 0.1)}


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


def save_head(path: Path, spec: dict, feature_size: int, state: dict, temperatures: dict, epoch: int, extra: dict) -> None:
    """Write one head and its metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": spec, "feature_size": feature_size, "state_dict": state, "temperatures": temperatures, "epoch": epoch, **extra}, path)


def export_winners(tag: str, size: str, output: Path, device: torch.device) -> None:
    """Retrain every winning configuration of one tag on all seeds, save the heads, and write the manifest."""
    train, names = load_split(FEATURE_DIR / f"{tag}_train_{size}.pt", device)
    validation, _ = load_split(FEATURE_DIR / f"{tag}_val_{size}.pt", device)
    test, _ = load_split(FEATURE_DIR / f"{tag}_test_{size}.pt", device)
    folder = output / "winners" / tag
    manifest = []
    for kind in KINDS:
        sweep = json.loads((MODELS_DIR / f"{tag}_{kind}_results.json").read_text(encoding="utf-8"))
        configs: dict = {}
        for group in GROUP_ORDER:
            if group in sweep["winners"]:
                key = json.dumps(sweep["winners"][group]["config"], sort_keys=True)
                configs.setdefault(key, {"config": sweep["winners"][group]["config"], "groups": [], "reported": sweep["winners"][group]["test"][MODE[kind]]["correctness_auroc"]["mean"]})
                configs[key]["groups"].append(group)
        for entry in configs.values():
            first = entry["groups"][0]
            aurocs, files = [], []
            for seed in SEEDS:
                spec = HeadSpec(**{**entry["config"], "seed": seed})
                run = train_head(spec, train, validation, test, names, device, keep_state=True)
                auroc = run["test"]["pooled"][MODE[kind]]["correctness_auroc"]
                path = folder / f"{kind}_{first}_seed{seed}.pt"
                save_head(path, run["spec"], int(train["prompt"].shape[1]), run["state"], run["temperatures"], run["epoch"],
                          {"kind": kind, "tag": tag, "groups": entry["groups"], "test_auroc": auroc, "datasets": names})
                aurocs.append(auroc)
                files.append(path.name)
            manifest.append({
                "kind": kind, "groups": entry["groups"], "files": files, "config": entry["config"],
                "test_auroc_per_seed": aurocs, "test_auroc_mean": float(np.mean(aurocs)), "sweep_reported_mean": entry["reported"],
                "difference": float(np.mean(aurocs) - entry["reported"]),
            })
            print(f"[saved] {tag} {kind} {'+'.join(entry['groups'])}: mean test AUROC {np.mean(aurocs):.4f} (sweep {entry['reported']:.4f})", flush=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def export_sequential(tag: str, kind: str, size: str, output: Path, device: torch.device) -> None:
    """Save the final head of each sequential-training variant and order (seed 1), with its own manifest."""
    train_all, names = load_split(FEATURE_DIR / f"{tag}_train_{size}.pt", device)
    validation_all, _ = load_split(FEATURE_DIR / f"{tag}_val_{size}.pt", device)
    test_all, _ = load_split(FEATURE_DIR / f"{tag}_test_{size}.pt", device)
    train, validation, test = (split_by_dataset(split, len(names)) for split in (train_all, validation_all, test_all))
    folder = output / "sequential" / tag
    manifest = []
    for variant, (loss, replay) in VARIANTS.items():
        config = load_config(kind, loss, MODELS_DIR, tag)
        for order_name, order_names in ORDER_NAMES.items():
            order = [names.index(n) for n in order_names]
            spec = HeadSpec(**{**config, "kind": kind, "seed": 1})
            run = run_sequence(spec, order, train, validation, test, device, replay, keep_head=True)
            final = run["stages"][-1]["scores"]
            aurocs = {names[d]: final[d][MODE[kind]]["auroc"] for d in range(len(names))}
            path = folder / f"{kind}_{variant}_{order_name}_seed1.pt"
            save_head(path, asdict(spec), int(train[0]["prompt"].shape[1]), run["head"]["state"], run["head"]["temperatures"],
                      run["head"]["epoch"], {"kind": kind, "tag": tag, "variant": variant, "order": order_names, "final_test_auroc": aurocs})
            manifest.append({"file": path.name, "kind": kind, "variant": variant, "order": order_names, "final_test_auroc": aurocs})
            print(f"[saved] {tag} {path.name}: final AUROC per dataset {aurocs}", flush=True)
    (folder / f"manifest_{kind}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Parse arguments and export the requested heads."""
    parser = argparse.ArgumentParser(description="Save the trained heads of one tag.")
    parser.add_argument("--mode", choices=("winners", "sequential"), required=True)
    parser.add_argument("--tag", required=True, help="mixture, mixture_noise, or <dataset>_<swap|noise>")
    parser.add_argument("--kind", choices=KINDS, help="Head kind, for sequential mode.")
    parser.add_argument("--size", default="2b")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "winners":
        export_winners(args.tag, args.size, args.output, device)
    else:
        if args.kind is None:
            parser.error("--kind is required for sequential mode")
        export_sequential(args.tag, args.kind, args.size, args.output, device)


if __name__ == "__main__":
    main()
