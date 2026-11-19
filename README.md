# VLM-LGT: Look, Ground, Then Think

Small vision-language models often answer confidently and wrongly. This project asks whether a frozen
Qwen3-VL can **notice when its answer is likely wrong** and **spend extra compute only there**, without fine-tuning the model:

1. **Detection.** A small head on the frozen model's hidden states predicts whether an answer is correct, far
   better than the model's own token probability (reports 01–09).
2. **Correction.** For flagged questions, generate candidate answers (zoomed and grounded crops, step-by-step
   reasoning, sampled reasoning, a larger model's answers) and let a trained ranker choose among them; a gate with a
   threshold fixed on validation decides which questions are worth it (reports 10–13).

Four benchmarks: ChartQA, TextVQA, DocVQA and ScienceQA (full test splits, 9,517 questions), plus AI2D and
InfographicVQA as datasets the pipeline never trained on. Every choice is made on validation and test is read
once; every result is compared with the model's greedy answer on the same questions (paired bootstrap, McNemar).

## Results

Qwen3-VL-2B, frozen. Test accuracy over the four datasets (greedy: 82.78%). Passes count generation calls plus the
scoring passes of each distinct candidate, with a 4B answer as 2 and an 8B answer as 4; seconds are measured on one
H100 at batch size 1 ([docs/pipeline.md](docs/pipeline.md#conventions), report 13 Sections 4 and 14).

| Method | Test accuracy | Gain (95% interval) | Passes | Seconds | Report |
|---|---:|---|---:|---:|---|
| Greedy answer | 82.78% | | 1 | 0.12 | |
| Best-of-7 samples, picked by the correctness head | 83.22% | +0.44 (+0.00 to +0.89) | 8 + scoring | | 10 |
| Ranker v1 over samples and crops | 83.85% | +1.07 (+0.77 to +1.40) | 13.5 | | 11 |
| Ranker v2 with reasoning and the model's own judge | 84.62% | +1.84 (+1.44 to +2.24) | 4.5 | | 11 |
| Ranker v3 (source agreement) + sampled reasoning | 86.38% | +3.60 (+3.10 to +4.09) | 12.0 | | 12 |
| … without the judge, gated at 50% | 86.38% | +3.60 (+3.10 to +4.10) | 5.8 | 2.45 | 13 |
| Route to Qwen3-VL-8B, no ranker (validation setting) | 88.29% | +5.52 (+4.93 to +6.10) | 3.6 | ≈0.22 | 13 |
| Ranker over the greedy and 8B answers, best under 0.3 s | 88.34% | +5.56 | 4.2 | 0.26 | 13 |
| Ranker v3 with the 8B answer and 8B reasoning as candidates, no judge | **89.53%** | **+6.76 (+6.10 to +7.37)** | 18.3 | 6.92 | 13 |
| … gated at 50% | 89.31% | +6.54 (+5.89 to +7.15) | 9.9 | 3.52 | 13 |

- **Ranker against routing:** below about 5 passes, routing the least-confident questions to the 8B model is as good
  as any ranker policy. The full ranker is above that routing by **+1.24 points (+0.78 to +1.68, p < 0.0001)** on the
  same questions.
- **What the ranker relies on:** source agreement. Removing it costs 4.8 points; removing the judge costs nothing and
  saves a scoring pass per candidate.
- **Robustness:** two further seed sets give 89.62% (89.53% originally). A ranker never trained on a dataset keeps
  85–100% of its gain there. 750 training questions per dataset already reach 89.20%.
- **Unseen datasets** (AI2D, InfographicVQA, frozen pipeline): the 2B-only ranker gains +3.85, as in-domain; the full
  system reaches 76.99% (+12.28), just above the 8B model on every question (76.52%).
- **Other base models:** Qwen3-VL-4B gains +1.71 from its own candidates and +3.54 with the 8B answers (89.38%);
  Qwen3-VL-8B gains +0.41.
- **Detection** (test AUROC for "the answer is correct"): unified head 0.86–0.93 per dataset, against 0.65–0.75 for
  the token probability (report 02).

**In progress:** fixes from other model families (InternVL3-8B, Idefics3-8B), and the 4B/8B base-model gating tables
(rerun without constant detectors).

## Reports

Self-contained HTML in [results/approach1/](results/approach1/), indexed in [results/README.md](results/README.md); the
data behind every number is in [results/approach1/04_sweep_data/](results/approach1/04_sweep_data/README.md).
Read 08 for the whole detection stage, 11 for the learned correction step, 12 for ranker v3 and gating, and 13 for
larger models, the accuracy–cost frontier and the current best result. Rendered online:
[report 13](https://raw.githack.com/arihanth26/VLM-LGT/main/results/approach1/13_escalation_report.html).

## Repository layout

```
VLM-LGT/
├── models/                     Qwen3-VL loading and inference (2B, 4B, 8B; InternVL3-8B, Idefics3-8B), token confidence
├── data/
│   ├── common/                 Shared dataset helpers
│   ├── approach1_latent_self_correction/   Dataset configs, downloads, splits (build_qhead_splits, build_unseen_splits)
│   └── approach2_verifier_reranking/       Approach 2 scaffold
├── experiments/
│   ├── approach1_latent_self_correction/   All Approach 1 code, grouped by stage in its README.md
│   │   ├── results/, q_features/, q_models/   Working outputs (only report inputs and head-sweep summaries tracked)
│   │   └── assets/             KaTeX for the HTML reports
│   └── approach2_verifier_reranking/       Approach 2 scaffold
├── slurm/                      Job scripts, *_pipeline.sh chains and batches/ (index: slurm/README.md)
├── tests/                      Unit tests (python -m tests.run_tests tests.<module>; no pytest needed)
├── trained_heads/approach1/    Exported correctness heads and probes
├── results/approach1/          Published reports 01–13 and their data (04_sweep_data/)
└── docs/                       pipeline.md (the correction pipeline, stage by stage), approach1_detailed_notes.md (detection stage)
```

Code goes by job, not by approach: `models/` loads and runs the model, `data/` builds datasets, `experiments/`
trains, evaluates and reports. Indexes: [experiments/approach1_latent_self_correction/README.md](experiments/approach1_latent_self_correction/README.md)
(every module, by stage), [slurm/README.md](slurm/README.md) (every job script and pipeline),
[docs/pipeline.md](docs/pipeline.md) (stages, outputs, conventions). See [CONTRIBUTING.md](CONTRIBUTING.md) and
[DATASETS.md](DATASETS.md).

## Getting started

```
pip install -r requirements.txt
python -m models.run_inference --size 2b --image path/to/image.jpg --question "What is in this image?"
```

On the ICE cluster (keep the repository, the Hugging Face cache and `.venv` in scratch; job scripts set `HF_HOME`
to `.hf_cache/` inside the repository):

```
sbatch slurm/setup_environment.sbatch                           # build .venv once
sbatch --export=ALL,MODEL_SIZE=8b slurm/prefetch_model.sbatch   # any key of models/model_config.py
bash slurm/check_scripts.sh                                     # every job script parses and its modules resolve
python -m tests.run_tests tests.test_gating                     # any test module; no network or GPU needed
```

## Reproducing the results

One command per stage chain, from the repository root on a login node:

```
bash slurm/run_correction_v3_pipeline.sh          # 2B candidates and ranker v3 (report 12)
bash slurm/run_escalation_pipeline.sh             # 4B/8B answers and routing (report 13)
bash slurm/run_escalation_ranker_pipeline.sh      # 8B answers as ranker candidates; ranker; gating
bash slurm/run_reasoning_8b_pipeline.sh           # 8B reasoning as a candidate
bash slurm/run_next_steps_pipeline.sh             # cascade; rankers without the judge
bash slurm/run_followups_pipeline.sh              # seeds, leave-one-dataset-out, training-set size, seconds
bash slurm/run_unseen_pipeline.sh                 # AI2D and InfographicVQA
SIZE=4b bash slurm/run_base_model_pipeline.sh     # another base model
python -m experiments.approach1_latent_self_correction.build_report_13   # rebuild report 13 from 04_sweep_data/
```

Every pipeline is described in [slurm/README.md](slurm/README.md); the detection-stage commands are in
[docs/approach1_detailed_notes.md](docs/approach1_detailed_notes.md). Reports 07, 12 and 13 rebuild from the
committed data alone (checked); regenerating candidates needs GPUs and the Qwen3-VL checkpoints.

## Branches

`main` holds everything that is merged and reproducible. Work happens on feature branches
(`feature/<topic>`, `<name>/<topic>`) that are merged into `main` when their results are in;
`feature/latent-self-correction` is the original Approach 1 branch and is kept as it is.

## Contributors

| Component | Contributor |
|---|---|
| Repository scaffold, Qwen3-VL loading and inference, dataset configs, Approach 2 scaffold | Arihanth Jayavijayan |
| Full-image baseline, token confidence, correctness heads, original ChartQA study, first correction experiments (100 examples) | Aryan Roy |
| Dataset extension, unified head, mixture and new loss, sequential training, full-split correction, candidate search, rankers v1–v2, gating, escalation and routing, rankers with larger-model candidates, baselines, robustness, transfer, base-model and other-family studies, latency | Arihanth Jayavijayan |
| Ranker v3 (source agreement), its dataset-input ablation, sampled reasoning, accuracy against cost | Mehdi Heydari |
