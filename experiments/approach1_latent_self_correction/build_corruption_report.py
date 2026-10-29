# build_corruption_report.py
#
# Builds the report for the corrupted-image version of the new loss as one
# self-contained HTML file. The loss is the same as in the swap version, but the
# corrupted state h_corrupt comes from the same image with noise mixed in instead of
# a different example's image. The report compares the two on the pooled mixture, on
# each dataset alone, and in sequential training, and checks whether grounding learned
# under one corruption carries over to the others.
# Reads q_models/mixture/*_results.json (written by sweep_mixture.py and sweep_continual.py)
# and reuses the charts and styling of build_report.py.

import json
import math
from pathlib import Path

from experiments.approach1_latent_self_correction.build_mixture_report import OLD_CSS, NEW_CSS, pair_plot
from experiments.approach1_latent_self_correction.build_report import CSS, esc, figure, table

HERE = Path(__file__).parent
MIXTURE_DIR = HERE / "q_models" / "mixture"
OUTPUT = HERE / "results" / "q_head_corruption_report.html"
KINDS = ("pre", "verifier", "unified")
MODE = {"pre": "main", "verifier": "main", "unified": "joint"}
KIND_LABEL = {"pre": "Separate pre-generation head", "verifier": "Separate verifier head", "unified": "Unified head (joint mode)"}
DATASETS = {"chartqa": "ChartQA", "textvqa": "TextVQA", "docvqa": "DocVQA", "scienceqa": "ScienceQA"}
CORRUPTIONS = ("swap", "noise", "mask", "shuffle")
VARIANTS = ("old_replay0", "old_replay0.1", "new_replay0", "new_replay0.1")
VARIANT_LABEL = {"old_replay0": "Old loss, no replay", "old_replay0.1": "Old loss, 10% replay",
                 "new_replay0": "New loss, no replay", "new_replay0.1": "New loss, 10% replay"}


def load(name: str):
    """One results file as a dict, or None if it was not produced."""
    path = MIXTURE_DIR / f"{name}_results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def auroc(entry: dict, kind: str) -> float:
    """Pooled test AUROC of a winner entry."""
    return entry["test"][MODE[kind]]["correctness_auroc"]["mean"]


def order_accuracy(node: dict) -> float:
    """Share of correct answers whose score falls under corruption, from a grounding node."""
    return node["correct"]["order_accuracy"]["mean"]


def own_grounding(entry: dict, kind: str) -> float:
    """Order accuracy against the corruption the head was trained or graded with."""
    return order_accuracy(entry["grounding"][MODE[kind]]["pooled"])


def extra_grounding(entry: dict, kind: str, corruption: str) -> float:
    """Order accuracy against another corruption of the test split."""
    return order_accuracy(entry["grounding_extra"][corruption][MODE[kind]]["pooled"])


def grounding_against(sweep: dict, group: str, kind: str, corruption: str) -> float:
    """Order accuracy of a winner group against one named corruption, whether it was the training one or an extra."""
    entry = sweep["winners"][group]
    if corruption == sweep["train_corruption"]:
        return own_grounding(entry, kind)
    return extra_grounding(entry, kind, corruption)


def pooled_rows(swap: dict, noise: dict) -> tuple[list, list]:
    """Table rows of pooled AUROC and own-corruption order accuracy for swap and noise, per head kind."""
    auroc_rows, ground_rows = [], []
    for kind in KINDS:
        s, n = swap[kind], noise[kind]
        auroc_rows.append([KIND_LABEL[kind], f"{auroc(s['winners']['old'], kind):.4f}", f"{auroc(s['winners']['new'], kind):.4f}",
                           f"{auroc(n['winners']['new'], kind):.4f}"])
        ground_rows.append([KIND_LABEL[kind], f"{grounding_against(n, 'old', kind, 'noise'):.3f}", f"{grounding_against(s, 'new', kind, 'noise'):.3f}",
                            f"{grounding_against(n, 'new', kind, 'noise'):.3f}"])
    return auroc_rows, ground_rows


def transfer_rows(swap: dict, noise: dict, kind: str) -> list:
    """Order accuracy against each corruption, for the old loss and the swap and noise versions of the new loss."""
    rows = []
    for label, sweep, group in (("Old loss", noise, "old"), ("New loss, trained with swap", swap, "new"), ("New loss, trained with noise", noise, "new")):
        rows.append([label] + [f"{grounding_against(sweep, group, kind, c):.3f}" for c in CORRUPTIONS])
    return rows


def individual_rows(kind: str) -> tuple[list, list]:
    """Per-dataset AUROC and own-corruption order accuracy for old loss, swap new loss and noise new loss."""
    auroc_rows, ground_rows = [], []
    for key, name in DATASETS.items():
        swap, noise = load(f"{key}_swap_{kind}"), load(f"{key}_noise_{kind}")
        if swap is None or noise is None:
            continue
        auroc_rows.append([name, f"{auroc(swap['winners']['old'], kind):.3f}", f"{auroc(swap['winners']['new'], kind):.3f}", f"{auroc(noise['winners']['new'], kind):.3f}"])
        ground_rows.append([name, f"{grounding_against(noise, 'old', kind, 'noise'):.3f}", f"{own_grounding(swap['winners']['new'], kind):.3f}",
                            f"{own_grounding(noise['winners']['new'], kind):.3f}"])
    return auroc_rows, ground_rows


def sequential_rows(kind: str) -> list:
    """Sequential-training summary for the swap and noise versions, per variant."""
    swap, noise = load(f"continual_{kind}") or {}, load(f"mixture_noise_continual_{kind}") or {}
    rows = []
    for variant in VARIANTS:
        for label, data in (("swap", swap), ("noise", noise)):
            if variant not in data.get("variants", {}):
                continue
            node = data["variants"][variant]
            summary = node["summary"][MODE[kind]]
            final = [entry["grounding_matrix"][MODE[kind]][-1] for entry in node["orders"].values()]
            ground = sum(sum(row) / len(row) for row in final) / len(final)
            rows.append([VARIANT_LABEL[variant], label, f"{summary['final_avg']['mean']:.3f}", f"{summary['forgetting']['mean']:.3f}", f"{ground:.3f}"])
    return rows


def sentence_ranges(values: list) -> str:
    """A 'low to high' phrase for a list of numbers."""
    return f"{min(values):.3f} to {max(values):.3f}"


def build() -> str:
    """Assemble the corruption report."""
    swap = {k: load(f"mixture_xeval_{k}") for k in KINDS}
    noise = {k: load(f"mixture_noise_{k}") for k in KINDS}
    base_swap = {k: load(f"mixture_{k}") for k in KINDS}
    if any(v is None for v in list(swap.values()) + list(noise.values())):
        raise SystemExit("Missing mixture_noise or mixture_xeval results. Run slurm/run_corruption_pipeline.sh first.")
    auroc_rows, ground_rows = pooled_rows(swap, noise)
    rows = noise["unified"]["rows"]

    old_auc = [auroc(noise[k]["winners"]["old"], k) for k in KINDS]
    new_auc = [auroc(noise[k]["winners"]["new"], k) for k in KINDS]
    swap_new_auc = [auroc(swap[k]["winners"]["new"], k) for k in KINDS]
    old_ground = [grounding_against(noise[k], "old", k, "noise") for k in KINDS]
    new_ground = [grounding_against(noise[k], "new", k, "noise") for k in KINDS]
    max_change = max(abs(a - b) for a, b in zip(new_auc, old_auc))
    transfer_swap_to_noise = [extra_grounding(swap[k]["winners"]["new"], k, "noise") for k in KINDS]
    transfer_noise_to_swap = [extra_grounding(noise[k]["winners"]["new"], k, "swap") for k in KINDS]

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q-Head Corruption Report</title><style>{CSS}</style></head><body><main>
<h1>The New Loss With a Corrupted Image Instead of a Swapped One</h1>
<p class="sub">The same loss as the swap version, with h_corrupt taken from the same image after noise is mixed in. Pooled mixture, each dataset alone, sequential training, and a check of how grounding carries over between corruptions.</p>
<p class="meta">Model: Qwen3-VL-2B. Mixture: {rows['train']:,} train, {rows['val']:,} validation, {rows['test']:,} test rows. Every number is on the test split, mean of three seeds.</p>"""]

    parts.append(f"""<h2 id="summary">1. In brief</h2><div class="card summary"><ul>
<li>The loss is unchanged. Only the source of h_corrupt changes: the question and the model's answer stay the same, and the image is replaced by itself mixed with Gaussian noise (alpha 0.15, so 15% of the signal variance is kept) instead of by another example's image.</li>
<li>Ranking quality is not affected. Pooled AUROC with the noise version of the new loss is within {max_change:.4f} of the old loss for every head kind, which is about the size of the seed-to-seed standard deviation (up to 0.0024).</li>
<li>Grounding rises. The share of correct answers whose score falls when the image is noised goes from {sentence_ranges(old_ground)} with the old loss to {sentence_ranges(new_ground)} with the noise version of the new loss (pre-generation, verifier, unified).</li>
<li>Grounding carries over between corruptions. Heads trained with noise score {sentence_ranges(transfer_noise_to_swap)} on swapped images they never trained on, and heads trained with swap score {sentence_ranges(transfer_swap_to_noise)} on noise. The old loss already scores about 0.88 to 0.95 on every corruption, so the gain is on top of a high floor.</li>
<li>Per dataset and in sequential training the picture matches the pooled one: AUROC unchanged, grounding higher. Forgetting with the noise loss is similar to the swap loss and is removed by replay in the same way.</li></ul></div>""")

    parts.append("""<h2 id="what">2. What changed and why</h2><div class="card"><p>The new loss adds a visual-grounding term to the correctness loss. For every correct answer the head scores the pair twice, once with the states of the clean image and once with the states of a corrupted image, and the term asks for the clean score to be higher:</p>
<p class="eq">L = L_cls + w_pre L_pre + w_pair L_pair + lambda L_ground, with L_ground = mean over correct answers of softplus(margin - (s(x_clean) - s(x_corrupt)))</p>
<p>The swap version builds x_corrupt from another example's image. That has a weakness: the corrupted pair is also a mismatched question and image, and a head can satisfy the term by detecting the mismatch, without looking at how well the image supports the answer. The noise version keeps the right image and removes detail, which takes away that shortcut. Noise follows the forward step of a diffusion process, x' = sqrt(a) x + sqrt(1 - a) eps with a = 0.15. Strength was chosen on 30 examples per dataset by how far the model's own answer probability falls on ScienceQA and TextVQA, so it is a reasonable setting, not a tuned optimum.</p>
<p>Mask (60% of a 16 by 16 grid blanked) and shuffle (4 by 4 patches permuted) were extracted for the test splits only. They are never trained on, so they work as a check that the learned grounding is not specific to one kind of corruption.</p></div>""")

    parts.append(f"""<h2 id="mixture">3. Pooled mixture</h2>
<p>All four datasets are pooled. Three head kinds, each with the old loss (lambda 0) and the new loss with lambda chosen on validation.</p>
<h3>Ranking quality (test AUROC)</h3>{table(["Head", "Old loss", "New loss, swap", "New loss, noise"], auroc_rows, left_cols=(0,))}
<p>The old loss column is the same head in both versions, so differences in the other two columns are the effect of the grounding term. All are within a few thousandths of the old loss.</p>
<h3>Grounding (share of correct answers whose score falls under noise)</h3>{table(["Head", "Old loss", "New loss, swap", "New loss, noise"], ground_rows, left_cols=(0,))}
<p>The noise column is graded on noise for all three. Even the old loss lowers its score on most noised images (a noised image changes the hidden state, and the head reads that state), so the starting point is high. The noise loss closes most of the remaining gap to 1.0 for all three heads.</p>""")

    parts.append("<h2 id=\"transfer\">4. Does grounding carry over to other corruptions?</h2><p>Each row is graded on the same test rows under four corruptions. Swap and noise are the two training choices, mask and shuffle are never trained on.</p>")
    for kind in KINDS:
        parts.append(f"<h3>{KIND_LABEL[kind]}</h3>" + table(["Trained with", "Swap", "Noise", "Mask", "Shuffle"], [[r[0], r[1], r[2], r[3], r[4]] for r in transfer_rows(swap[kind], noise[kind], kind)], left_cols=(0,)))
    plot_rows = []
    for kind in KINDS:
        for corruption in ("mask", "shuffle"):
            plot_rows.append((f"{KIND_LABEL[kind].split(' (')[0]}, {corruption}", grounding_against(noise[kind], "old", kind, corruption),
                              grounding_against(noise[kind], "new", kind, corruption), None))
    parts.append(figure("fig-transfer", "Grounding on corruptions that were never trained on", pair_plot(plot_rows, 0.85, 1.0, "Order accuracy (old loss, new loss trained with noise)", "fig-transfer"),
                        "Orange is the old loss and blue the noise-trained new loss. Both are graded on mask and shuffle, which neither saw in training."))
    parts.append("""<p>Training with either corruption raises grounding on every other corruption compared with the old loss. For heads trained with swap, the gain over the old loss is about 0.03 to 0.04 on noise and mask and about 0.015 to 0.024 on shuffle. For heads trained with noise, the gain is largest on mask (0.05 to 0.06), which is expected since both remove detail instead of changing content, and smallest on swap (0.02 to 0.03). Because both training choices improve all four, the gain is not just a detector for one corruption type, although all four corruptions still alter the hidden state in ways a head can read.</p>""")

    parts.append("<h2 id=\"single\">5. Each dataset alone</h2><p>Each dataset gets its own heads, with the old loss, the swap new loss and the noise new loss.</p>")
    for kind in KINDS:
        a, g = individual_rows(kind)
        parts.append(f"<h3>{KIND_LABEL[kind]}</h3><p>Test AUROC</p>" + table(["Dataset", "Old loss", "New loss, swap", "New loss, noise"], a, left_cols=(0,))
                     + "<p>Order accuracy under noise (noise column), swap (swap column) or own corruption</p>" + table(["Dataset", "Old loss (noise)", "New loss, swap (swap)", "New loss, noise (noise)"], g, left_cols=(0,)))
    parts.append("""<p>On ChartQA, TextVQA and DocVQA the old loss already lowers the score on about 0.97 or more of correct answers under noise, so there is little left for the grounding term to add and AUROC does not move. ScienceQA is where the term matters: the old loss lowers the score on only 0.69 to 0.78 of correct answers under noise, and the noise loss raises that to 0.87 to 1.0 depending on the head. ScienceQA questions can often be answered from the text alone, so a score that ignores the image is the expected outcome without the term, and ScienceQA AUROC changes by less than 0.01 either way.</p>""")

    parts.append("<h2 id=\"seq\">6. Sequential training and forgetting</h2><p>Heads are trained on one dataset after another (three orders, three seeds) and every dataset is scored after every stage. Forgetting is the average drop from the best earlier score on a dataset to its score at the end. Last column is order accuracy after the final stage under the version's own corruption.</p>")
    for kind in KINDS:
        parts.append(f"<h3>{KIND_LABEL[kind]}</h3>" + table(["Variant", "Corruption", "Final mean AUROC", "Forgetting", "Grounding after last stage"], sequential_rows(kind), left_cols=(0, 1)))
    parts.append("""<p>The old-loss rows are the same runs in both versions (the corruption only changes how grounding is graded). With the noise loss, grounding after the last stage stays high (0.97 to 1.0 with replay for the pre-generation and verifier heads), so the grounding is retained as datasets are added. Forgetting without replay is higher for the noise loss on the pre-generation head (0.025 against 0.013) and the verifier head (0.046 against 0.033) and about the same on the unified head (0.013 against 0.011). With 10% replay the noise-loss values are 0.016, 0.008 and 0.014, close to the old loss, so the same remedy works.</p>""")

    parts.append("""<h2 id="limits">7. How far to trust this</h2><div class="card"><ul>
<li>Order accuracy near 1.0 is partly easy. A noised image moves the hidden state a lot, so a head can learn to lower its score on noisy-looking inputs without checking whether the image supports the answer. The cross-corruption check lowers this worry but does not remove it. A stronger test is a corruption that removes the evidence for the answer while keeping the image natural, such as masking only the region the answer comes from, which we did not run.</li>
<li>The corruption strength was set on a 30-example smoke test from the model's own answer probability, and only one strength was trained.</li>
<li>The second term of the loss (the reference answer must outscore the wrong predicted answer) is our reading of the contrastive term and is the same in both versions. An embedding-level contrastive loss would need a change to the head loss.</li>
<li>Mask and shuffle were only used for grading, so they show transfer, not training under those corruptions.</li>
<li>Grounding here means the score falls when the image is degraded. It does not show that the score tracks the specific visual evidence for an answer.</li>
<li>AUROC differences between old and new loss are inside the seed spread, so this version improves grounding and does not improve ranking.</li></ul></div>""")
    parts.append("</main></body></html>")
    return "".join(parts)


def main() -> None:
    """Build the report and write it next to the other analysis outputs."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"[report] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
