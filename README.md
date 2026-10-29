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
| Approach 1: unified head that reads both latents in one network | Done, compared against the separate heads on all four datasets |
| Approach 1: ChartQA rerun under the same protocol as the new datasets | Done, reproduces the original study numbers |
| Approach 1: all four datasets pooled into one mixture, with the model's own confidence and linear probes tested on it | Done |
| Approach 1: separate and unified heads on the mixture, old loss | Done |
| Approach 1: new loss with a visual-grounding term, separate and unified heads on the mixture | Done |
| Approach 1: sequential training (train on A, then B, then C) to test what an earlier dataset keeps, old and new loss, with and without replay | Done |
| Approach 1: new loss with a noised image instead of a swapped one as the corrupted input, on the mixture, each dataset alone, and sequential training | Done |
| Approach 1: prompted retry on answers flagged by the unified head | Implemented, experiment not yet run |
| Approach 1: a learned latent correction step that acts on a flagged answer | Not started |
| Approach 1: Qwen3-VL 4B check | Not started |
| Approach 2: verifier reranking | Scaffold only: dataset configs and a failure-pool builder with a placeholder sampler |

The detection half of Approach 1 is in good shape. A small head trained on frozen Qwen3-VL
hidden states predicts whether an answer is correct much better than the model's own token
confidence, and that holds on charts, scene text, documents, and science diagrams. What is
missing is the second half: a correction step that uses this score to fix flagged answers.

### Results so far

Test correctness AUROC for Qwen3-VL-2B (0.5 is chance). Token confidence is the geometric
mean of the answer token probabilities. The unified head is shown in its joint mode, where it
sees the generated answer. The full write-up opens with a summary of the original ChartQA
study and its numbers, then covers the datasets with example images, per-dataset results with
case studies, calibration, selective-prediction curves, ablations, interpretation, and
limitations. It is in
[results/approach1/02_multidataset_qhead_report.html](results/approach1/02_multidataset_qhead_report.html).
All published result files are indexed in [results/README.md](results/README.md), numbered in
the order the work happened, starting from the original ChartQA study.

| Dataset | Input type | Base accuracy | Token confidence | Best separate head | Unified head |
|---|---|---:|---:|---:|---:|
| ChartQA (rerun, same protocol) | Charts | 75.8% | 0.751 | 0.895 | 0.898 |
| TextVQA | Scene text | 81.0% | 0.726 | 0.868 | 0.867 |
| DocVQA | Scanned documents | 87.0% | 0.685 | 0.788 | 0.811 |
| ScienceQA (image questions) | Science diagrams | 86.7% | 0.654 | 0.918 | 0.925 |

Short version of what the numbers say:

- The learned heads beat token confidence on every dataset, by roughly 0.1 to 0.26 AUROC.
- The unified head ties the separate heads on ChartQA and TextVQA and beats them on DocVQA and
  ScienceQA, using one model instead of two. Paired bootstrap intervals are in the report.
- Flagging the 20% lowest-scored answers captures 59% to 79% of all errors with the unified
  head, against 44% to 53% with token confidence.
- The ChartQA rerun reproduces the original study: token confidence AUROC 0.751 against 0.750,
  and the linear pre-generation probe 0.895 against 0.895.
- The pairwise reference-answer loss hurt on every dataset and is off in every winning
  configuration. The extra pre-generation loss term helps the pre-generation mode at almost no
  cost to the joint mode.
- These results are about detection. Whether a correction step can fix the flagged answers is
  the next experiment.

Training across datasets (report
[results/approach1/05_mixture_new_loss_report.html](results/approach1/05_mixture_new_loss_report.html)):

- One unified head trained on the pooled mixture reaches pooled AUROC 0.881, against 0.719 for
  the untrained answer-token probability. On each dataset it is within 0.015 AUROC of the head trained on
  that dataset alone, and above it on three of the four.
- Trained on one dataset after another, heads lose little of what they learned earlier:
  forgetting is 0.011 AUROC for the unified head and 0.013 for the pre-generation head. The
  answer-conditioned verifier forgets the most (0.032), and a replay buffer holding 10% of
  earlier training rows cuts that to 0.005.
- The new loss adds a term that makes the score fall when the image is replaced by a corrupted
  one. That term alone costs no accuracy and raises how often the score falls for correct
  answers (unified head: 93% to 96%). The other contrastive term (predicted against correct
  answer) lowers AUROC by 0.01 to 0.02, so the loss as first specified, with both terms, is
  slightly worse than the old loss.
- The answer-token probability (no training) already falls for 95% of correct answers under a
  corrupted image. Its weakness is ranking right against wrong answers, not noticing missing
  evidence.

Trained heads: all 540 trained correctness heads (every sweep winner on three seeds for the pooled mixture, the noise
mixture and each dataset alone, plus the final heads of the sequential runs) are in
[trained_heads/approach1/](trained_heads/approach1/), with a guide to loading them and how they were checked in
[trained_heads/README.md](trained_heads/README.md).

Corrupted image instead of a swapped one (report
[results/approach1/06_corruption_new_loss_report.html](results/approach1/06_corruption_new_loss_report.html)):

- The corrupted input of the grounding term is the same image with Gaussian noise mixed in,
  which removes the shortcut of detecting a mismatched question and image. The loss is otherwise
  unchanged.
- Pooled AUROC does not move (within 0.0024 of the old loss for every head). The share of
  correct answers whose score falls under noise rises from 0.92 to 0.94 with the old loss to
  0.97 to 1.00 with the noise loss.
- Grounding carries over to corruptions never trained on: heads trained with noise reach 0.91 to
  0.97 on swapped images, and both versions improve on mask and shuffle over the old loss.
- Each dataset alone and sequential training match the pooled picture. The gain is clearest on
  ScienceQA, where the old loss lowers the score on only 0.69 to 0.78 of correct answers under
  noise. Forgetting is similar to the swap version, and replay removes it in the same way.

### Contributors

| Component | Scope | Contributor |
|---|---|---|
| Project foundation | Repository scaffold, Qwen3-VL loading and inference, dataset configs with Hub verification, Approach 2 scaffold | Arihanth Jayavijayan |
| Baseline and confidence | Full-image baseline, token-likelihood confidence, baseline reports | Aryan Roy |
| Correctness heads | Frozen feature extraction, pre-generation head, answer-conditioned verifier head, sweeps, original ChartQA study | Aryan Roy |
| Dataset extension | TextVQA, DocVQA, and ScienceQA selection, grouped splits, per-dataset scoring | Arihanth Jayavijayan |
| Unified head | Combined head and loss, seeded sweep, comparison against the separate heads | Arihanth Jayavijayan |

## Repo layout

Three top-level folders split the code by job, not by approach: `models/` only loads and
runs Qwen3-VL, `data/` only acquires and preprocesses datasets, `experiments/` is
everything that trains, sweeps, scores, or reports on top of that data. Each of `data/`
and `experiments/` then splits again by approach. See
[CONTRIBUTING.md](CONTRIBUTING.md#where-code-goes) for the reasoning and for where to put
new code.

```
VLM-LGT/
├── models/                                    Loading and running Qwen3-VL (2B and 4B), shared by both approaches
│   ├── model_config.py                        Registry of model sizes and their Hugging Face repo ids
│   ├── load_model.py                          Loads a model and processor for a given size
│   ├── run_inference.py                       Runs the model on an image and a question, with token confidence
│   ├── inference_result.py                    Result record for one generation (answer plus confidence values)
│   └── uncertainty.py                         Summarizes token log probabilities into confidence values
│
├── data/                                       Dataset acquisition and preprocessing, nothing else
│   ├── common/
│   │   ├── utils.py                           Shared helpers used by both approaches (jsonl read/write, etc)
│   │   └── dataset_entry.py                   Shared DatasetEntry structure (repo id, config, source, notes)
│   │
│   ├── approach1_latent_self_correction/      Approach 1 datasets
│   │   ├── dataset_config.py                  Dataset groups: training, testbed, ood, and the Q-head datasets
│   │   ├── download.py                        Downloads the raw datasets
│   │   ├── preprocess.py                      Converts raw data into a shared training format
│   │   ├── build_qhead_splits.py              Builds train/val/test splits for TextVQA, DocVQA, ScienceQA
│   │   ├── raw/                               Downloaded or built datasets (not tracked in git)
│   │   └── processed/                         Preprocessed jsonl output (not tracked in git)
│   │
│   └── approach2_verifier_reranking/          Approach 2 datasets
│       ├── README.md                          Start here if you are working on Approach 2
│       ├── dataset_config.py
│       ├── download.py
│       ├── preprocess.py
│       ├── raw/
│       └── processed/
│
├── experiments/                                Training, sweeps, analysis, and reports, no dataset acquisition
│   ├── approach1_latent_self_correction/      Approach 1: baseline, correctness heads, analysis, reports
│   │   ├── baseline.py                        Full-image generation over a split, with per-answer correctness
│   │   ├── scoring.py                         Per-dataset answer metrics (VQA accuracy, ANLS, option letter)
│   │   ├── report.py                          Markdown accuracy report for a baseline run
│   │   ├── extract_q_features.py              Caches frozen Qwen3-VL states for the correctness heads
│   │   ├── corruption.py                      Image corruptions (swap, patch shuffle, noise, mask) for the grounding loss
│   │   ├── extract_corrupt_features.py        Caches states and answer confidence under a corrupted image
│   │   ├── build_mixture.py                   Pools the cached features of all datasets into one mixture
│   │   ├── mixture_heads.py                   Old and new loss for separate and unified heads, pooled and
│   │   │                                      per-dataset evaluation, grounding metrics
│   │   ├── sweep_mixture.py                   Seeded sweeps of the three head kinds on the mixture
│   │   ├── analyze_mixture.py                 Baselines, curves, and the clean-versus-corrupted gallery
│   │   ├── continual.py                       Sequential training across datasets and the retention metrics
│   │   ├── sweep_continual.py                 Dataset orders, old and new loss, replay, isolated references
│   │   ├── build_mixture_report.py            Builds the mixture, sequential, and new-loss HTML report
│   │   ├── export_heads.py                    Saves the trained heads (sweep winners, sequential) and loads them back
│   │   ├── build_corruption_report.py         Builds the report on the noise version of the new loss
│   │   ├── build_approach1_report.py          Builds the complete Approach 1 report (HTML, math embedded, PDF export)
│   │   ├── export_results_workbook.py         Writes every results table to one Excel workbook
│   │   ├── train_q_heads.py                   Pre-generation head and verifier head, shared losses and metrics
│   │   ├── sweep_q_heads.py                   Architecture and hyperparameter sweep for the separate heads
│   │   ├── sweep_verifier_early_stop.py       Early-stopping sweep over the verifier ranking weight
│   │   ├── unified_q_head.py                  Unified head over both latents, with its combined loss
│   │   ├── sweep_unified_q_head.py            Seeded sweep of the unified head, winner scored once on test
│   │   ├── analyze_q_heads.py                 Curves, reliability, categories, case studies, bootstrap intervals
│   │   ├── compare_q_heads_report.py          Markdown table comparing all scorers per dataset
│   │   ├── build_report.py                    Builds the multi-dataset HTML report from result files
│   │   ├── publish_results.py                 Copies results into the numbered top-level results/ folder
│   │   ├── q_features/                        Cached hidden states (not tracked in git)
│   │   ├── q_models/                          Sweep outputs per dataset: sweep_results.json and
│   │   │                                      unified_sweep_results.json (checkpoints not tracked)
│   │   └── results/                           Baseline outputs, analysis files, and the HTML reports
│   │       ├── confidence_comparison_report.html       Original ChartQA study report
│   │       ├── q_head_multidataset_report.html         Multi-dataset report with charts
│   │       ├── q_head_comparison.md                    Side-by-side metric tables
│   │       └── analysis_<dataset>.json                 Curves and intervals behind the report
│   │
│   └── approach2_verifier_reranking/          Approach 2: policy sampling, verifier, and evaluation code
│       ├── build_failure_pool.py              Builds the verifier's failure-inclusive training pool
│       └── failure_pool/                      Output of the above (not tracked in git)
│                                               Everything else (verifier model, training, reranking,
│                                               evaluation) is not written yet, add it here
│
├── results/                                   Published results, numbered in the order the work happened
│   ├── README.md                              Index of every result file and what stage it comes from
│   ├── approach1/                             Approach 1 results (working copies stay under experiments/)
│   │   ├── 01_chartqa_original_study_report.html   First study: ChartQA only
│   │   ├── 02_multidataset_qhead_report.html       Three more datasets and the unified head
│   │   ├── 03_qhead_comparison_tables.md           Side-by-side metric tables
│   │   ├── 04_sweep_data/                          Sweep and analysis JSON files per dataset and for the mixture
│   │   ├── 05_mixture_new_loss_report.html         Mixture, sequential training, and the new loss
│   │   ├── 06_corruption_new_loss_report.html      The new loss with a noised image instead of a swapped one
│   │   ├── 07_correction_experiments_report.html   Prompted retry, reranking, best-of-five, deeper pass
│   │   ├── 08_approach1_complete_report.html       Everything in Approach 1, also as a PDF, for the mid-point report
│   │   └── 08_approach1_results.xlsx               Every results table as one Excel workbook, one sheet per table
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
│   ├── analyze_q_heads.sbatch                 CPU job: curves, case studies, and bootstrap intervals
│   ├── extract_corrupt_features.sbatch        Caches states under a corrupted image on a GPU
│   ├── build_mixture.sbatch                   CPU job: pools the datasets into the mixture
│   ├── sweep_mixture.sbatch                   Sweeps one head kind on the mixture on a GPU
│   ├── analyze_mixture.sbatch                 CPU job: baselines and curves for the mixture
│   ├── run_qhead_pipeline.sh                  Submits the whole chain for one dataset with dependencies
│   ├── sweep_continual.sbatch                 Sequential training for one head kind on a GPU
│   ├── run_mixture_pipeline.sh                Submits the mixture and new-loss chain for all datasets
│   ├── run_continual_pipeline.sh              Submits the sequential-training jobs, one per head kind
│   ├── export_heads.sbatch                    Retrains and saves the heads of one tag to trained_heads/
│   ├── run_export_pipeline.sh                 Submits the jobs that save every trained head
│   ├── run_corruption_pipeline.sh             Submits the noise-corruption chain: features, builds, sweeps, sequential runs
│   └── check_scripts.sh                       Validates every job script: syntax, module paths, sbatch --test-only
│
├── trained_heads/
│   └── approach1/                             All 540 trained correctness heads (winners and sequential) and a loading guide

├── docs/
│   └── approach1_unified_q_head.md            Design note: datasets, unified head, loss, protocol, status
│
├── tests/
│   ├── test_config.py                         Smoke tests for the model and dataset configs, no network needed
│   ├── test_scoring_and_splits.py             Offline tests for answer scoring and the grouped split
│   ├── test_unified_q_head.py                 Offline tests for the unified head and its sweep
│   ├── test_mixture.py                        Offline tests for corruptions, the old and new loss, and the mixture sweep
│   ├── test_continual.py                      Offline tests for sequential training and the retention metrics
│   ├── test_mixture_report.py                 Offline test that the mixture report builds from the real result schema
│   ├── test_corruption_report.py              Offline test that the corruption report builds from the sweep schema
│   └── test_approach1_report.py               Offline tests for the complete report builder helpers
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
   python -m tests.test_mixture
   python -m tests.test_continual
   python -m tests.test_mixture_report
   python -m tests.test_corruption_report
   python -m tests.test_approach1_report
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
python -m experiments.approach1_latent_self_correction.analyze_q_heads --dataset textvqa
python -m experiments.approach1_latent_self_correction.compare_q_heads_report
python -m experiments.approach1_latent_self_correction.build_report
python -m experiments.approach1_latent_self_correction.publish_results   # refresh the copies in results/approach1
```

Dataset choice and the reasons other candidates were rejected, the unified head and its loss,
the selection protocol, and the current status are written up in
[docs/approach1_unified_q_head.md](docs/approach1_unified_q_head.md).

### Dataset mixture and the new loss

The second set of experiments pools all four datasets into one mixture (the union of their
train, validation, and test splits) and trains and tests the heads on the pool. It needs the
clean features and baselines from the per-dataset pipeline above. One command submits the whole
chain:

```
bash slurm/run_mixture_pipeline.sh
```

It runs, in order: corrupted-image features for every dataset and split (GPU, in parallel),
the mixture build (CPU), one sweep per head kind (pre-generation, verifier, unified) under the
old and the new loss (GPU, in parallel), and the analysis (CPU). The sequential-training
experiments (train on one dataset, then the next, and measure what is forgotten) run on the
same cached features and use the winning configuration of each mixture sweep:

```
PRE_JOB=<id> VERIFIER_JOB=<id> UNIFIED_JOB=<id> bash slurm/run_continual_pipeline.sh
```

Pass the job ids of the mixture sweeps to start each sequential job when its sweep finishes,
or run it with no ids once the sweeps are done. Then build the report and publish it:

```
python -m experiments.approach1_latent_self_correction.build_mixture_report
python -m experiments.approach1_latent_self_correction.publish_results
```

The corrupted-image version of the new loss (noise mixed into the same image instead of a swapped
image) runs the same chain for the noise features, the pooled and per-dataset builds, all sweeps,
and the sequential runs. The cluster allows 50 submitted jobs per user, so the script can be run a
second time with `SKIP_POOLED=1 SINGLES="scienceqa"` for whatever did not fit:

```
bash slurm/run_corruption_pipeline.sh
python -m experiments.approach1_latent_self_correction.build_corruption_report
```

The new loss adds a visual-grounding term: the score of the clean input must exceed the score of the same question
and answer with a corrupted image. The design is in
[docs/approach1_unified_q_head.md](docs/approach1_unified_q_head.md).

### Prompted retry on the mixture test set

The prompted-retry benchmark generates baseline answers, extracts and scores their states online,
then generates one visually grounded retry for every test example. Dataset downloads and Arrow
caches use the compute node's temporary disk; no baseline or feature tensors are kept in the repo.
The same retry outputs
are used to compare six policies without generation noise between conditions: no retry, retry all,
head confidence below 0.5, the lowest 20% by head confidence, the lowest 20% by token confidence,
and a seeded random 20% control. The output is resumable and the summary reports coverage, fixed
errors, broken answers, and final accuracy for every policy. It also scores the retry with the same
unified head and compares direct candidate reranking, confidence-gated replacement at margins of
0, 0.05, 0.10, and 0.20, and the oracle upper bound where either candidate can be selected.

```
sbatch slurm/run_prompted_retry.sbatch
```

Set `REVIEW_FRACTION`, `THRESHOLD`, or `RANDOM_SEED` to change the comparison policies. Set
`LIMIT=100` for a small end-to-end check before submitting the full test mixture.

The best-of-five experiment samples five independent candidates at temperature 0.9 and top-p
0.95, scores each with the unified head, and compares head selection with the first sample,
majority vote, token-confidence selection, the greedy baseline, and an oracle upper bound. It
writes JSONL, summary JSON, and a self-contained HTML report:

```
LIMIT=100 sbatch slurm/run_candidate_reranking.sbatch
```

Check every job script before a long run. This validates shell syntax, that each `python -m`
module resolves, and, on the cluster, that the scheduler accepts the resource request:

```
bash slurm/check_scripts.sh
```

Jobs request realistic time limits on purpose. The cluster caps the total GPU minutes a user
can have requested at once, so asking for 8 hours on a 20 minute job leaves other jobs waiting.

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
