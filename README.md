# VLM-LGT: Look, Ground, Then Think

Small vision-language models often answer confidently and wrongly. This project asks whether a frozen
Qwen3-VL can **notice when its answer is likely wrong** and **fix it at test time**, without fine-tuning the model:

1. **Detection.** A small head on the frozen model's hidden states predicts whether an answer is correct, far
   better than the model's own token probability (reports 01–09).
2. **Correction.** For flagged questions, generate better candidate answers (zoomed and grounded crops,
   step-by-step reasoning, sampled reasoning, a larger model) and let a trained ranker choose among them; a gate
   decides which questions are worth the extra compute (reports 10–13).

Four benchmarks: ChartQA, TextVQA, DocVQA and ScienceQA (full test splits, 9,517 questions), plus AI2D and
InfographicVQA as datasets the pipeline never trained on. Every choice is made on validation and test is read
once; every result is compared with the model's greedy answer on the same questions (paired bootstrap, McNemar).

## Results

Qwen3-VL-2B, frozen. Test accuracy over the four datasets (greedy: 82.78%). Model passes count generation calls
plus the scoring passes of each distinct candidate (docs/pipeline.md, Conventions); the 8B row counts one 8B answer as 4.

| Method | Test accuracy | Gain (95% interval) | Model passes per question | Report |
|---|---:|---|---:|---|
| Greedy answer | 82.78% | | 1 | |
| Best-of-7 samples (T = 0.7), picked by the correctness head | 83.22% | +0.44 (+0.00 to +0.89), p = 0.052 | 8 + scoring | 10 |
| Ranker v1 over samples and crops | 83.85% | +1.07 (+0.77 to +1.40) | 13.5 | 11 |
| Ranker v2 with reasoning and the model's own judge | 84.62% | +1.84 (+1.44 to +2.24) | 4.5 | 11 |
| Ranker v3 (one ranker over every correction source) | 85.87% | +3.09 (+2.54 to +3.61) | 24.1 | 12 |
| Ranker v3 + sampled reasoning | **86.38%** | **+3.60 (+3.10 to +4.09)** | 12.0 | 12 |
| … gated: corrects the 50% least-confident questions | 86.38% | +3.60 (+3.10 to +4.10) | **7.0** | 12 |
| Route the 30% least-confident questions to Qwen3-VL-8B | 87.32% | +4.54 (+4.01 to +5.08) | ≈2.1 | 13 |

Detection (test AUROC for "the answer is correct"): unified head 0.86–0.93 per dataset against 0.65–0.75 for the
token probability (report 02). Experiments in progress (report 13): the 8B answers and 8B reasoning as ranker
candidates, training-free baselines at matched cost, ranker input ablations, Qwen3-VL-4B as the base model,
transfer to unseen datasets, and measured latency.

## Reports

Self-contained HTML in [results/approach1/](results/approach1/), indexed in [results/README.md](results/README.md).
Read 08 for the whole detection stage, 11 for the learned correction step, 12 for the current best ranker and gating.
The full pipeline, stage by stage, is in [docs/pipeline.md](docs/pipeline.md).

## Repository layout

```
VLM-LGT/
├── models/                              Qwen3-VL loading and inference (2B, 4B, 8B), token confidence
├── data/
│   ├── common/                          Shared dataset helpers
│   ├── approach1_latent_self_correction/   Dataset configs, downloads, splits (build_qhead_splits, build_unseen_splits)
│   └── approach2_verifier_reranking/    Approach 2 scaffold
├── experiments/approach1_latent_self_correction/
│   ├── baseline.py, scoring.py          Greedy baseline and per-dataset metrics (VQA accuracy, ANLS, option letter)
│   ├── extract_q_features.py, …         Frozen hidden states; correctness heads, sweeps, mixture, new loss (reports 01–09)
│   ├── candidate_pools.py               Greedy + sampled answers per question (report 10)
│   ├── correction_candidates.py         Round 1: contrastive decoding, zoomed crops, candidate encoding (report 11)
│   ├── correction_v2.py, correction_v3_cot.py   Rounds 2–3: grounded crops, reasoning, judge, sampled reasoning
│   ├── escalation.py, escalation_candidates.py  Larger models, upscaled images, 8B reasoning; as ranker candidates
│   ├── train_candidate_ranker.py, train_ranker_v2.py, train_ranker_v3.py   The candidate rankers
│   ├── gating.py, gate_ranker_v3.py     Deployable gating (validation thresholds) of a saved ranker
│   ├── analyze_*.py, compute_baselines.py, evaluate_unseen.py, measure_latency.py   Analyses
│   ├── build_*report*.py                Report builders
│   ├── correction_scale.py              Shared constants: datasets, splits, base model switch, shards
│   ├── q_features/, q_models/, results/ Working outputs (not tracked; published copies are in results/)
├── slurm/                               One job script per stage, and *_pipeline.sh dependency chains
├── tests/                               Unit tests (run with python -m tests.run_tests tests.<module>)
├── trained_heads/approach1/             Exported correctness heads and probes
├── results/approach1/                   Published reports (01–12) and their data (04_sweep_data/)
└── docs/                                pipeline.md (the correction pipeline), approach1_detailed_notes.md (earlier stages and commands)
```

Code goes by job, not by approach: `models/` loads and runs the model, `data/` builds datasets, `experiments/`
trains, evaluates and reports. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Getting started

```
pip install -r requirements.txt
python -m models.run_inference --size 2b --image path/to/image.jpg --question "What is in this image?"
```

On the ICE cluster (keep the repository, the Hugging Face cache and `.venv` in scratch; job scripts set `HF_HOME`
to `.hf_cache/` inside the repository):

```
sbatch slurm/setup_environment.sbatch              # build .venv once
sbatch slurm/prefetch_model.sbatch                 # MODEL_SIZE=4b or 8b for the larger models
bash slurm/check_scripts.sh                        # every job script parses and its modules resolve
python -m tests.run_tests tests.test_gating        # any test module; no network or GPU needed
```

Then one command per stage chain, for example `bash slurm/run_escalation_ranker_pipeline.sh`; the full list, with
what each runs and where it writes, is in [docs/pipeline.md](docs/pipeline.md#pipelines-one-command-submits-a-whole-dependency-chain).
The commands of the detection stage are in [docs/approach1_detailed_notes.md](docs/approach1_detailed_notes.md).

## Branches

`main` holds everything that is merged and reproducible. Work happens on feature branches
(`feature/<topic>`, `<name>/<topic>`) that are merged into `main` when their results are in;
`feature/latent-self-correction` is the original Approach 1 branch and is kept as it is.

## Contributors

| Component | Contributor |
|---|---|
| Repository scaffold, Qwen3-VL loading and inference, dataset configs, Approach 2 scaffold | Arihanth Jayavijayan |
| Full-image baseline, token confidence, correctness heads, original ChartQA study, first correction experiments (100 examples) | Aryan Roy |
| Dataset extension, unified head, mixture and new loss, sequential training, full-split correction, candidate search, rankers v1–v2, gating, escalation, transfer and base-model studies | Arihanth Jayavijayan |
| Ranker v3 (source agreement), its dataset-input ablation, sampled reasoning, accuracy against cost | Mehdi Heydari |
