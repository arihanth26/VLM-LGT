# Slurm job scripts (PACE ICE)

Submit everything from the repository root. Job scripts (`*.sbatch`) run one stage; pipelines (`run_*_pipeline.sh`) submit
a whole dependency chain from a login node. GPU scripts request the `coc-gpu,ice-gpu` partitions (16 h limit) and set
`HF_HOME` to `.hf_cache/` in the repository. The cluster allows 50 queued jobs and a limited number of GPU-minutes per
user, counted on the requested time, so time limits are set close to the measured run times.

```
sbatch slurm/setup_environment.sbatch                       # build .venv once
sbatch --export=ALL,MODEL_SIZE=8b slurm/prefetch_model.sbatch   # any key of models/model_config.py
bash slurm/check_scripts.sh                                 # every script parses and its modules resolve
```

## Pipelines

| Command | Runs | Report |
|---|---|---|
| `run_qhead_pipeline.sh`, `run_mixture_pipeline.sh`, `run_corruption_pipeline.sh`, `run_continual_pipeline.sh`, `run_export_pipeline.sh` | Detection heads: per dataset, mixture, corruptions, sequential, export | 02–09 |
| `run_correction_full_pipeline.sh` | First correction experiments on the full test split | 07–08 |
| `run_search_pipeline.sh`, `run_extend_pipeline.sh` | Candidate pools and the candidate search | 10 |
| `run_correction_learning_pipeline.sh`, `run_correction_v2_pipeline.sh` | Rounds 1–2 and rankers v1–v2 | 11 |
| `run_correction_v3_pipeline.sh` | Round 3 (sampled reasoning) and ranker v3 | 12 |
| `run_escalation_pipeline.sh` | 4B, 8B and enlarged-image answers; routing | 13 |
| `run_escalation_ranker_pipeline.sh`, `run_reasoning_8b_pipeline.sh` | 8B answers and 8B reasoning as ranker candidates; ranker; gating | 13 |
| `run_next_steps_pipeline.sh` | Cascade; rankers without the judge | 13 |
| `run_followups_pipeline.sh` | Seeds, leave-one-dataset-out, training-set size, gating in seconds; 8B reasoning on the unseen datasets | 13 |
| `run_unseen_pipeline.sh` | AI2D and InfographicVQA through the frozen pipeline | 13 |
| `run_base_model_pipeline.sh` (`SIZE=4b` or `8b`), `run_base_model_escalation_pipeline.sh` | The whole pipeline with another base model; with the 8B answers | 13 |
| `run_families_pipeline.sh` | InternVL3-8B and Idefics3-8B as fixes and ranker candidates; latency on A100/L40S | in progress |

## Batches: several short runs in one job

`sbatch --export=ALL,BATCH=<file> slurm/run_batch.sbatch` runs every line of `slurm/batches/<file>` (a module and its
options) in order, after the unit tests, so several experiments use one queue slot. A failed run is logged and the next
one still runs.

| Batch files | Runs |
|---|---|
| `headline_seeds`, `headline_holdout`, `data_efficiency`, `nojudge_seeds` | Seeds, leave-one-dataset-out and training-set size of the rankers |
| `headline_ablations`, `headline_variants` | Input ablations and variants of the best ranker |
| `seconds_gating` | Gating with measured seconds |
| `best_ranker_analysis`, `best_ranker_analysis_4b` | Paired comparisons and answer origins (2B, 4B base model) |
| `esc8b_candidates_<dataset>` | The 8B answers encoded by another base model |
| `fix_<model>_<dataset>`, `fix_families_unseen`, `encode_families_<dataset>`, `families_ranker`, `routing_other_families` | Other-family fixes |
| `rerun_detectors_4b`, `rerun_detectors_8b` | 4B/8B gating and routing without constant detectors |

## Stage scripts

| Scripts | Stage |
|---|---|
| `build_qhead_splits.sbatch`, `download_approach1.sbatch` | Datasets |
| `extract_q_features.sbatch`, `extract_corrupt_features.sbatch` | Frozen hidden states |
| `train_q_heads.sbatch`, `sweep_*.sbatch`, `build_mixture.sbatch`, `export_heads.sbatch`, `analyze_q_heads.sbatch`, `analyze_mixture.sbatch` | Detection heads |
| `run_baseline.sbatch`, `run_prompted_retry.sbatch`, `run_candidate_reranking.sbatch`, `run_deepen_selected.sbatch`, `run_correction_shard.sbatch`, `merge_correction_shards.sbatch` | First correction experiments |
| `run_candidate_pool.sbatch`, `run_extend_pools.sbatch`, `analyze_candidate_pools.sbatch` | Candidate pools |
| `run_correction_candidates.sbatch`, `run_correction_v2.sbatch`, `run_correction_v3_cot.sbatch` | Correction rounds 1–3 |
| `train_candidate_ranker.sbatch`, `train_ranker_v2.sbatch`, `train_ranker_v3.sbatch`, `analyze_ranker.sbatch` | Rankers |
| `gate_ranker_v3.sbatch` | Gating |
| `run_escalation.sbatch`, `run_reasoning_8b.sbatch`, `run_escalation_candidates.sbatch`, `analyze_escalation.sbatch`, `analyze_cascade.sbatch` | Larger models |
| `compute_baselines.sbatch`, `measure_latency.sbatch` | Baselines, latency |
| `run_unseen_step.sbatch`, `run_base_model_chain.sbatch` | Unseen datasets; another base model |
| `run_batch.sbatch` | Batches |
