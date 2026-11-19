# Approach 1 code: detect the frozen model's errors, then correct them

Every module runs as `python -m experiments.approach1_latent_self_correction.<module>` from the repository root, on an
ICE GPU node through its job script in [`slurm/`](../../slurm/README.md). The whole correction pipeline, stage by stage,
is in [`docs/pipeline.md`](../../docs/pipeline.md); the reports built from these modules are indexed in
[`results/README.md`](../../results/README.md).

| Folder | What it holds | Tracked |
|---|---|---|
| `*.py` (this folder) | The code, grouped below | yes |
| `results/` | Working outputs of every stage (per-shard JSONL, hidden states, rankers); published copies are in `results/approach1/04_sweep_data/` | seven report inputs only |
| `q_features/` | Cached frozen hidden states per dataset split | no |
| `q_models/` | Sweep summaries of the detection heads | summaries only |
| `assets/` | KaTeX for the self-contained HTML reports | yes |

## Shared building blocks

| Module | Role |
|---|---|
| `correction_scale.py` | Datasets, splits, shards, the base-model switch (`VLMLGT_MODEL`), result folders |
| `scoring.py` | Per-dataset answer scoring (relaxed accuracy, VQA accuracy, ANLS, option letter) |
| `correction_stats.py` | Paired bootstrap intervals, McNemar tests, AUROC, per-dataset rows |
| `gating.py` | Validation thresholds and per-question gate decisions |
| `baseline.py`, `report.py` | Greedy baseline run and its accuracy report |

## Detection: correctness heads (reports 01–09)

| Module | Role |
|---|---|
| `extract_q_features.py`, `extract_prompt_features.py`, `extract_corrupt_features.py` | Frozen hidden states for the heads (clean, question-only, corrupted image) |
| `train_q_heads.py`, `sweep_q_heads.py`, `sweep_verifier_early_stop.py` | Pre-generation and answer heads per dataset |
| `unified_q_head.py`, `sweep_unified_q_head.py`, `sweep_pre_head.py` | One head over both latents; its sweeps |
| `build_mixture.py`, `mixture_heads.py`, `sweep_mixture.py`, `corruption.py` | Heads trained on the dataset mixture, with the visual-grounding loss |
| `continual.py`, `sweep_continual.py` | Sequential training across datasets |
| `export_heads.py`, `heads_vs_probes.py`, `compare_heads_vs_linear.py` | Final heads and probes (`trained_heads/approach1/`), heads against linear probes |
| `analyze_q_heads.py`, `analyze_mixture.py`, `compare_q_heads_report.py` | Analyses behind reports 02–06 |

## Correction: candidates, rankers, gating (reports 07–12)

| Module | Role |
|---|---|
| `prompted_retry.py`, `online_prompted_retry.py`, `candidate_reranking.py`, `deepen_selected_candidate.py`, `merge_correction_shards.py` | First correction experiments: retry, best-of-five, deeper answers (report 07) |
| `candidate_pools.py`, `extend_pools.py`, `analyze_candidate_pools.py`, `selection_metrics.py` | Greedy + sampled answers per question, the candidate search (report 10) |
| `correction_candidates.py` | Round 1: contrastive decoding, zoomed crops; encodes every candidate |
| `correction_v2.py`, `repair_v2_shards.py` | Round 2: grounded crops, reasoning, the model's own judge |
| `correction_v3_cot.py` | Round 3: three sampled reasoning answers |
| `train_candidate_ranker.py`, `train_ranker_v2.py`, `train_ranker_v3.py` | Rankers v1–v3 (v3: source agreement; `--drop`, `--hold-out`, `--seed-offset`, `--train-limit`) |
| `gate_ranker_v3.py` | Gating of a saved ranker (validation thresholds, passes and seconds, reproduction check) |
| `analyze_ranker.py` | Analysis behind report 11 |

## Larger models and further evidence (report 13)

| Module | Role |
|---|---|
| `escalation.py` | Answers of a larger or other-family model (4B, 8B, InternVL3-8B, Idefics3-8B), 8B reasoning, enlarged images |
| `escalation_candidates.py` | Those answers encoded as ranker candidates (`correction_v4`–`v6`) |
| `analyze_escalation.py`, `analyze_cascade.py` | Routing to a larger model; the 2B → 4B → 8B cascade |
| `compute_baselines.py` | Training-free baselines on the same candidates |
| `analyze_best_ranker.py` | Paired comparisons, answer origins, oracle gap, abstention, examples |
| `evaluate_unseen.py` | The frozen pipeline on AI2D and InfographicVQA |
| `measure_latency.py` | Seconds per component on one GPU |

## Reports

| Module | Builds |
|---|---|
| `build_report.py`, `build_mixture_report.py`, `build_corruption_report.py` | Reports 02, 05, 06 |
| `build_correction_report.py` | Report 07 |
| `build_approach1_report.py`, `export_results_workbook.py` | Report 08 and its workbook |
| `build_search_report.py`, `build_ranker_report.py`, `build_report_12.py`, `build_report_13.py` | Reports 10–13 |
| `publish_results.py` | Copies working results into `results/approach1/04_sweep_data/` |

Reports 07, 08, 12 and 13 read the published copies in `results/approach1/04_sweep_data/`; reports 07, 12 and 13 were
checked to rebuild from a fresh clone (report 08's builder also needs `openpyxl` for its workbook).
Reports 02 and 05 also read seven files kept in `results/` for that reason (`analysis_*.json`, the original ChartQA report
and its validation runs). Reports 10 and 11 read working outputs that are not committed; their published data is in
`04_sweep_data/` (`candidate_search.json`, `candidate_ranker.json`, `ranker_analysis.json`).
