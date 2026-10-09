# VLM-LGT: Look, Ground, Then Think

This repo holds the code for our project on teaching vision-language models to notice when
their visual evidence is weak and do something about it before answering, instead of just
guessing.

We are testing two ways to do this:

1. **Latent Self-Correction.** The model scores its own uncertainty and, when it is not
   confident, updates its own internal representation and re-answers, without necessarily
   fetching new pixels.
2. **Verifier Reranking.** The model produces several candidate answers and a separately
   trained verifier picks the best one.

Both approaches build on Qwen3-VL and share the same broad evaluation goals, but the code for
each one lives in its own folder so they can be developed independently.

## Where we stand

| Area | Status |
|---|---|
| Model loading and inference for Qwen3-VL 2B and 4B | Done |
| Dataset configs, download, and preprocessing for both approaches | Done, ids verified against the Hub |
| Approach 1: full-image baseline with token-likelihood confidence | Done |
| Approach 1: correctness heads on frozen features (pre-generation head and answer-conditioned verifier), sweeps, ChartQA study | Done |
| Approach 1: same experiment on TextVQA, DocVQA, and ScienceQA | Done, run on the ICE cluster |
| Approach 1: unified head that reads both latents in one network | Done, compared against the separate heads on all three new datasets |
| Approach 1: ChartQA rerun under the same protocol as the new datasets | Running on ICE, the report regenerates when it finishes |
| Approach 1: the latent correction step that acts on a flagged answer | Not started |
| Approach 1: counterfactual control and Qwen3-VL 4B check | Not started |
| Approach 2: verifier reranking | Scaffold only: dataset configs and a failure-pool builder with a placeholder sampler |

The detection half of Approach 1 is in good shape. A small head trained on frozen Qwen3-VL
hidden states predicts whether an answer is correct much better than the model's own token
confidence, and that holds on charts, scene text, documents, and science diagrams. What is
missing is the second half: a correction step that uses this score to fix flagged answers.

### Results so far

Test correctness AUROC for Qwen3-VL-2B (0.5 is chance). Token confidence is the geometric
mean of the answer token probabilities. The unified head is shown in its joint mode, where it
sees the generated answer. The full write-up, with charts, calibration, selective-prediction
curves, ablations, and limitations, is in
[results/approach1/02_multidataset_qhead_report.html](results/approach1/02_multidataset_qhead_report.html).
All published result files are indexed in [results/README.md](results/README.md), numbered in
the order the work happened, starting from the original ChartQA study.

| Dataset | Input type | Base accuracy | Token confidence | Best separate head | Unified head |
|---|---|---:|---:|---:|---:|
| ChartQA (original study) | Charts | 75.8% | 0.750 | 0.895 | not run yet |
| TextVQA | Scene text | 81.0% | 0.726 | 0.868 | 0.867 |
| DocVQA | Scanned documents | 87.0% | 0.685 | 0.788 | 0.811 |
| ScienceQA (image questions) | Science diagrams | 86.7% | 0.654 | 0.918 | 0.925 |

Short version of what the numbers say:

- The learned heads beat token confidence on every dataset, by roughly 0.1 to 0.26 AUROC.
- The unified head ties the separate heads on TextVQA and beats them on DocVQA and ScienceQA,
  using one model instead of two. Paired bootstrap intervals are in the report.
- Flagging the 20% lowest-scored answers captures 59% to 79% of all errors with the unified
  head, against 44% to 53% with token confidence.
- The pairwise reference-answer loss hurt on every dataset and is off in every winning
  configuration. The extra pre-generation loss term helps the pre-generation mode at almost no
  cost to the joint mode.
- These results are about detection. Whether a correction step can fix the flagged answers is
  the next experiment.

### Contributors

| Component | Scope | Contributor |
|---|---|---|
| Project foundation | Repository scaffold, Qwen3-VL loading and inference, dataset configs with Hub verification, Approach 2 scaffold | Arihanth Jayavijayan |
| Baseline and confidence | Full-image baseline, token-likelihood confidence, baseline reports | Aryan Roy |
| Correctness heads | Frozen feature extraction, pre-generation head, answer-conditioned verifier head, sweeps, original ChartQA study | Aryan Roy |
| Dataset extension | TextVQA, DocVQA, and ScienceQA selection, grouped splits, per-dataset scoring | Arihanth Jayavijayan |
| Unified head | Combined head and loss, seeded sweep, comparison against the separate heads | Arihanth Jayavijayan |
| Pipeline and reporting | ICE job chain, analysis tooling, multi-dataset report, numbered results folder | Arihanth Jayavijayan |

## Repo layout

```
VLM-LGT/
├── models/                                    Loading and running Qwen3-VL (2B and 4B)
│   ├── model_config.py                        Registry of model sizes and their Hugging Face repo ids
│   ├── load_model.py                          Loads a model and processor for a given size
│   ├── run_inference.py                       Runs the model on an image and a question, with token confidence
│   ├── inference_result.py                    Result record for one generation (answer plus confidence values)
│   └── uncertainty.py                         Summarizes token log probabilities into confidence values
│
├── data/
│   ├── common/
│   │   ├── utils.py                           Shared helpers used by both approaches (jsonl read/write, etc)
│   │   └── dataset_entry.py                   Shared DatasetEntry structure (repo id, config, source, notes)
│   │
│   ├── approach1_latent_self_correction/      Approach 1: datasets, baseline, correctness heads, analysis
│   │   ├── dataset_config.py                  Dataset groups: training, testbed, ood, and the Q-head datasets
│   │   ├── download.py                        Downloads the raw datasets
│   │   ├── preprocess.py                      Converts raw data into a shared training format
│   │   ├── baseline.py                        Full-image generation over a split, with per-answer correctness
│   │   ├── scoring.py                         Per-dataset answer metrics (VQA accuracy, ANLS, option letter)
│   │   ├── build_qhead_splits.py              Builds train/val/test splits for TextVQA, DocVQA, ScienceQA
│   │   ├── report.py                          Markdown accuracy report for a baseline run
│   │   ├── extract_q_features.py              Caches frozen Qwen3-VL states for the correctness heads
│   │   ├── train_q_heads.py                   Pre-generation head and verifier head, shared losses and metrics
│   │   ├── sweep_q_heads.py                   Architecture and hyperparameter sweep for the separate heads
│   │   ├── sweep_verifier_early_stop.py       Early-stopping sweep over the verifier ranking weight
│   │   ├── unified_q_head.py                  Unified head over both latents, with its combined loss
│   │   ├── sweep_unified_q_head.py            Seeded sweep of the unified head, winner scored once on test
│   │   ├── analyze_q_heads.py                 Curves, reliability, categories, and bootstrap intervals
│   │   ├── compare_q_heads_report.py          Markdown table comparing all scorers per dataset
│   │   ├── build_report.py                    Builds the multi-dataset HTML report from result files
│   │   ├── publish_results.py                 Copies results into the numbered top-level results/ folder
│   │   ├── raw/                               Downloaded or built datasets (not tracked in git)
│   │   ├── processed/                         Preprocessed jsonl output (not tracked in git)
│   │   ├── q_features/                        Cached hidden states (not tracked in git)
│   │   ├── q_models/                          Sweep outputs per dataset: sweep_results.json and
│   │   │                                      unified_sweep_results.json (checkpoints not tracked)
│   │   └── results/                           Baseline outputs, analysis files, and the HTML reports
│   │       ├── confidence_comparison_report.html       Original ChartQA study report
│   │       ├── q_head_multidataset_report.html         Multi-dataset report with charts
│   │       ├── q_head_comparison.md                    Side-by-side metric tables
│   │       └── analysis_<dataset>.json                 Curves and intervals behind the report
│   │
│   └── approach2_verifier_reranking/          Approach 2: datasets for the verifier reranking approach
│       ├── dataset_config.py
│       ├── download.py
│       ├── preprocess.py
│       ├── build_failure_pool.py              Builds the failure-inclusive pool the verifier trains on
│       ├── raw/
│       ├── processed/
│       └── failure_pool/
│
├── results/                                   Published results, numbered in the order the work happened
│   ├── README.md                              Index of every result file and what stage it comes from
│   ├── approach1/                             Approach 1 results (working copies stay under data/)
│   │   ├── 01_chartqa_original_study_report.html   First study: ChartQA only
│   │   ├── 02_multidataset_qhead_report.html       Three more datasets and the unified head
│   │   ├── 03_qhead_comparison_tables.md           Side-by-side metric tables
│   │   └── 04_sweep_data/                          Sweep and analysis JSON files per dataset
│   └── approach2/                             Approach 2 results (none yet)
│
├── slurm/                                     Job scripts for the ICE cluster
│   ├── setup_environment.sbatch               Creates the project virtual environment on a CPU node
│   ├── prefetch_model.sbatch                  Downloads the Qwen3-VL weights once
│   ├── download_approach1.sbatch              Downloads Approach 1 datasets
│   ├── build_qhead_splits.sbatch              Builds the TextVQA, DocVQA, or ScienceQA splits
│   ├── run_baseline.sbatch                    Full-image baseline on a GPU
│   ├── extract_q_features.sbatch              Caches frozen states on a GPU
│   ├── train_q_heads.sbatch                   Trains the two separate heads once
│   ├── sweep_q_heads.sbatch                   Sweeps the separate heads
│   ├── sweep_verifier_early_stop.sbatch       Early-stopping verifier sweep
│   ├── sweep_unified_q_head.sbatch            Sweeps the unified head
│   └── run_qhead_pipeline.sh                  Submits the whole chain for one dataset with dependencies
│
├── docs/
│   └── approach1_unified_q_head.md            Design note: datasets, unified head, loss, protocol, status
│
├── tests/
│   ├── test_config.py                         Smoke tests for the model and dataset configs, no network needed
│   ├── test_scoring_and_splits.py             Offline tests for answer scoring and the grouped split
│   └── test_unified_q_head.py                 Offline tests for the unified head and its sweep
│
├── requirements.txt
├── DATASETS.md                                Reference table of every dataset id, with source links and known quirks
├── CONTRIBUTING.md                            How to add a model, add a dataset, and code style expectations
└── README.md
```

## Getting started

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Try the model:

   ```
   python -m models.run_inference --size 2b --image path/to/image.jpg --question "What is in this image?"
   ```

   Swap `--size 2b` for `--size 4b` to compare the larger checkpoint on the same input. Both
   sizes share the same loading code in `models/load_model.py`, so anyone on the project can
   switch between them without changing anything else.

3. Pull and prepare data for an approach:

   ```
   python -m data.approach1_latent_self_correction.download
   python -m data.approach1_latent_self_correction.preprocess
   ```

   The same pattern applies under `data/approach2_verifier_reranking/`.

4. Check that the code is still sane before a long run. None of these touch the network or
   need a GPU:

   ```
   python -m tests.test_config
   python -m tests.test_scoring_and_splits
   python -m tests.test_unified_q_head
   ```

## Approach 1 on the ICE cluster

Run everything from the repository root and keep the repo, the Hugging Face cache, and the
environment in scratch, because the home quota is 30 GB. The job scripts set `HF_HOME` to a
cache folder inside the repo for this reason.

### One-time setup

```
sbatch slurm/setup_environment.sbatch      # builds .venv on a CPU node
sbatch slurm/prefetch_model.sbatch         # downloads the Qwen3-VL weights once
```

Wait for each job to finish before starting the next step.

### Full-image baseline

Approach 1 first measures the unmodified Qwen3-VL model over a complete dataset split. Each
example receives exactly one full-image generation. Token likelihood is logged for calibration
analysis but never selects or replaces an answer.

```
sbatch --export=ALL,DATASET=chartqa,DATA_GROUP=ood slurm/download_approach1.sbatch
sbatch --export=ALL,DATASET=chartqa,DATA_GROUP=ood,SPLIT=test,MODEL_SIZE=2b,MAX_NEW_TOKENS=512 \
    slurm/run_baseline.sbatch
```

The output defaults to `chartqa_test_2b_baseline.jsonl` plus a readable Markdown report in the
Approach 1 `results/` directory. ChartQA uses relaxed accuracy: numeric answers within 5% of
the reference count as correct, while text answers require normalized exact matching. The other
datasets use their own metrics from `scoring.py`. Output is appended per example, so a
repeated job resumes missing indices after interruption. GPU jobs request a flexible
constraint (`H200|H100|L40S|A100-80GB|A100-40GB|A40`) rather than a fixed card, since the
account's partitions do not include H200s.

### Correctness heads on one dataset

One command per dataset submits the whole chain with job dependencies: build or download the
splits, baseline generation and feature extraction for train, validation, and test, then the
separate-head sweep and the unified-head sweep.

```
bash slurm/run_qhead_pipeline.sh textvqa          # or docvqa, scienceqa_img, chartqa_full
```

When the sweeps finish, produce the analysis and reports:

```
python -m data.approach1_latent_self_correction.analyze_q_heads --dataset textvqa
python -m data.approach1_latent_self_correction.compare_q_heads_report
python -m data.approach1_latent_self_correction.build_report
python -m data.approach1_latent_self_correction.publish_results   # refresh the copies in results/approach1
```

Dataset choice and the reasons other candidates were rejected, the unified head and its loss,
the selection protocol, and the current status are written up in
[docs/approach1_unified_q_head.md](docs/approach1_unified_q_head.md).

## Working with both model sizes

The project standardizes on two Qwen3-VL checkpoints, registered in
`models/model_config.py`:

- `2b` (`Qwen/Qwen3-VL-2B-Instruct`), good for fast local iteration when you are testing an
  idea or debugging a script.
- `4b` (`Qwen/Qwen3-VL-4B-Instruct`), closer to what full experiments will actually run on.

Everyone should default to `2b` while developing, then confirm results hold on `4b` before
reporting them. All results so far are on `2b`. Both sizes go through the same
`load_qwen3_vl(size)` function, so any script that takes a `--size` flag works with either one
without code changes:

```
python -m models.run_inference --size 2b --image example.jpg --question "What is happening here?"
python -m models.run_inference --size 4b --image example.jpg --question "What is happening here?"
```

Run both on the same input when you want to check whether a behavior is specific to model
scale before you build on top of it.

## Notes

- Model loading defaults to bfloat16 and picks a GPU automatically if one is available,
  falling back to CPU otherwise.
- Every dataset id in `dataset_config.py` was checked against the live Hugging Face Hub.
  See [DATASETS.md](DATASETS.md) for the full table, the verification dates, and known
  quirks (split naming, required configs, file formats). If `download.py` errors on a
  dataset, check that file before assuming the code is wrong, dataset owners do rename
  and restructure repos over time. The `lmms-lab` org currently redirects to
  `lmms-lab-encoder`, which `load_dataset` follows.
- TextVQA and DocVQA do not release test answers, so their val and test splits are carved out
  of the labeled validation split by image or document. They are not the official test sets.
- Ref-Adv-S, used in the Approach 1 testbed, is not on the Hugging Face Hub and needs to
  be downloaded manually from its GitHub repo. `download.py` will tell you if a local
  copy is missing. Ref-L4 is on the Hub and downloads automatically.
- If you hit Hugging Face rate limits, log in once with `huggingface-cli login`.
- `build_failure_pool.py` under Approach 2 is a starting scaffold. The exact candidate
  sampling format is still an open decision, so the sampling function is left as a
  placeholder until that is settled.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a model size, add a dataset, and
the code style this repo follows. Short version: if you are experimenting with a
different model size or dataset, add it to the relevant config file rather than
hardcoding paths in a script, so everyone else picks it up automatically.
