# The correction pipeline, stage by stage

Everything below runs on a frozen Qwen3-VL (2B unless stated), on the ICE cluster, from the repository root. Each
stage is a Python module under `experiments/approach1_latent_self_correction/` (written `E.` below) with a Slurm job
script under `slurm/`. Every stage writes per-shard JSONL (+ a `.pt` of hidden states where it encodes answers) under
`experiments/approach1_latent_self_correction/results/` (written `results/`), resumes from what it already wrote, and
records failures per question instead of stopping. Choices (configuration, detector, budget, learning rate, epoch)
are always made on validation; test is read once.

## Overview

```
                  ┌──────────── detection (reports 01–09) ────────────┐
 image + question │ greedy answer ─► correctness head / probe / token │
                  └───────────────────────────┬───────────────────────┘
                                              ▼
                 gate: correct this question?  (threshold fixed on validation)
                      │ no ─► keep the greedy answer
                      ▼ yes
   candidates: samples · zoomed crops · contrastive decoding · grounded crops · reasoning ·
               sampled reasoning · Qwen3-VL-8B answer · 8B reasoning · upscaled image
                                              ▼
   encode every distinct candidate with the 2B model on the full image:
   hidden state · answer probability · head and probe score · the model's own yes/no judge
                                              ▼
   ranker v3 (scores each candidate; the top one is the answer)
```

## Stages

| # | Stage | Module | Job script | Output | Report |
|---|---|---|---|---|---|
| 0 | Datasets: ChartQA, TextVQA, DocVQA, ScienceQA (train/val/test) | `data/.../build_qhead_splits.py` | `build_qhead_splits.sbatch` | `data/.../raw/<dataset>/` | 02 |
| 0b | Unseen datasets: AI2D, InfographicVQA (split `unseen`) | `data/.../build_unseen_splits.py` | `run_unseen_step.sbatch` (STEP=build) | `data/.../raw/{ai2d,infovqa}/` | 13 |
| 1 | Question embedding (the constant input of every head score) | `E.extract_q_features`, `E.extract_prompt_features` | `extract_q_features.sbatch`, `run_unseen_step.sbatch` | `E/q_features/<dataset>_<split>_<size>.pt` | 02 |
| 2 | Correctness heads and probes | `E.sweep_mixture`, `E.export_heads` | `sweep_mixture.sbatch`, `export_heads.sbatch` | `trained_heads/approach1/` | 05–09 |
| 3 | Pools: greedy + 10 samples at T = 0.7/1.0/1.3, every answer scored | `E.candidate_pools` | `run_candidate_pool.sbatch` | `results/pools/T<t>/<split>/` | 10 |
| 4 | Round 1: contrastive decoding (2), zoomed crops (4); encode every distinct candidate | `E.correction_candidates` | `run_correction_candidates.sbatch` | `results/correction/<split>/` | 11 |
| 5 | Round 2: grounded crop (box, crop, crop + context), reasoning, judge for every candidate | `E.correction_v2` | `run_correction_v2.sbatch` | `results/correction_v2/<split>/` | 11 |
| 6 | Round 3: 3 sampled reasoning answers | `E.correction_v3_cot` | `run_correction_v3_cot.sbatch` | `results/correction_v3/<split>/` | 12 |
| 7 | Escalation answers: 8B, 4B, 2B on images enlarged 1.5x/2x, 8B reasoning | `E.escalation` | `run_escalation.sbatch`, `run_reasoning_8b.sbatch` | `results/escalation/{model_8b,model_4b,upscale,reasoning_8b}/` | 13 |
| 8 | Escalation answers as ranker candidates (encoded by 2B, judged) | `E.escalation_candidates` | `run_escalation_candidates.sbatch` | `results/correction_v4/` (8B, upscaled), `results/correction_v5/` (8B reasoning) | 13 |
| 9 | Rankers: v1/v2 (report 11), v3 with source agreement (and `--with-cot-samples`, `--with-escalation`, `--with-reasoning-8b`, `--drop <inputs>`, `--hold-out <dataset>` for leave-one-dataset-out) | `E.train_candidate_ranker`, `E.train_ranker_v2`, `E.train_ranker_v3` | `train_*.sbatch` | `results/candidate_ranker*.json`, `*_models.pt` | 11–13 |
| 10 | Gating of a saved ranker (validation thresholds; ungated rows must reproduce the ranker's run; `--drop judge --scoring-passes 1` for a ranker without the judge) | `E.gating`, `E.gate_ranker_v3` | `gate_ranker_v3.sbatch` | `results/gating_v3*.json` | 12–13 |
| 11 | Routing to a larger model / upscaled image (no ranker) | `E.analyze_escalation` | `analyze_escalation.sbatch` | `results/escalation.json` | 13 |
| 11b | Cascade 2B -> 4B -> 8B against routing straight to 8B, at matched cost | `E.analyze_cascade` | `analyze_cascade.sbatch` | `results/cascade.json` | 13 |
| 12 | Training-free baselines on the same candidates | `E.compute_baselines` | `compute_baselines.sbatch` | `results/baselines.json` | 13 |
| 13 | Transfer to unseen datasets (frozen pipeline) | `E.evaluate_unseen` | `run_unseen_pipeline.sh` | `results/unseen.json` | 13 |
| 14 | Latency of every component on one H100 | `E.measure_latency` | `measure_latency.sbatch` | `results/latency.json` | 13 |
| 15 | Another base model: the whole pipeline with Qwen3-VL-4B (`VLMLGT_MODEL=4b`) | stages 1, 3–6, 9, 10 | `run_base_model_pipeline.sh` | `results/base_4b/` | 13 |

## Pipelines (one command submits a whole dependency chain)

| Command | What it runs |
|---|---|
| `bash slurm/run_search_pipeline.sh` | Stage 3 for val/test at three temperatures, train pass rates, candidate search, pre-generation head sweep (report 10) |
| `bash slurm/run_correction_learning_pipeline.sh` | Stages 4 and 9 (ranker v1), report 11 analysis |
| `bash slurm/run_correction_v2_pipeline.sh` | Stage 5 and ranker v2 |
| `bash slurm/run_correction_v3_pipeline.sh` | Stage 6 and ranker v3 with sampled reasoning |
| `bash slurm/run_escalation_pipeline.sh` | Stage 7 (8B, upscaled) for val/test and stage 11 |
| `bash slurm/run_escalation_ranker_pipeline.sh` | Stage 7 on the train subset, stage 8, ranker v3 with the escalation candidates, stage 10 |
| `bash slurm/run_reasoning_8b_pipeline.sh` | 8B reasoning (stages 7–8), ranker v3 with it, stage 10 |
| `bash slurm/run_unseen_pipeline.sh` | Stages 1, 3–8 on the unseen split, then stage 13 |
| `bash slurm/run_base_model_pipeline.sh` | Stage 15 (two jobs per dataset split; `DATASETS=`, `SPLITS=`, `RANKER=0` to submit only some chains) |
| `bash slurm/run_next_steps_pipeline.sh` | Stage 11b; rankers without the judge (one scoring pass per candidate), with and without the 8B candidates, and their gating |
| `bash slurm/run_followups_pipeline.sh` | Packed follow-ups of the best ranker (seeds, leave-one-dataset-out, training-set size, gating in seconds) and 8B reasoning on the unseen datasets with a second unseen evaluation |
| `sbatch --export=ALL,BATCH=<file> slurm/run_batch.sbatch` | Several short runs in one job (one per line of `slurm/batches/<file>`), to stay under the 50-job queue limit |

## Conventions

- **Question sets.** Validation and test are the full splits (6,420 and 9,517 questions). Rankers train on 12,000
  training questions (3,000 per dataset) that every correction round covers (`correction_scale.ranker_subset`).
- **Cost.** Generation calls per question follow `train_ranker_v3.generation_calls` (greedy, each sample, each source;
  the grounded crops share one box call; an 8B answer counts 4, a 4B answer 2). Passes add one encoding and one judge
  pass per distinct candidate (only the encoding pass for a ranker trained without the judge). Measured seconds are in `results/latency.json`.
  With `results/latency.json`, gating also reports estimated seconds per question on one H100 (`gate_ranker_v3.SECONDS_OF`).
- **Statistics.** Paired bootstrap intervals (2,000 resamples) and exact McNemar tests against the greedy answer on the
  same questions (`E.correction_stats`).
- **Another base model.** `VLMLGT_MODEL=4b` switches every stage to that model and every result folder to
  `results/base_4b/`; the 2B heads cannot read its states, so their scores are a constant 0.5. `VLMLGT_CHAIN=1` makes
  every stage shard the same questions, so one job can run all stages for its shard.
- **Tests.** `python -m tests.run_tests tests.<module> ...` runs a test module without pytest (the cluster
  environment has none); the gating, ranker, escalation, baseline and transfer tests run on a GPU node before every
  analysis job.
