# Results

Published results for the project, numbered in the order the work happened. Working copies
stay where the code writes them (`experiments/approach1_latent_self_correction/results/` and
`.../q_models/`). The files here are copies, refreshed with:

```
python -m experiments.approach1_latent_self_correction.publish_results
```

## Approach 1: latent self-correction

| File | Stage | What it shows |
|---|---|---|
| [01_chartqa_original_study_report.html](approach1/01_chartqa_original_study_report.html) | First study, ChartQA only | A learned correctness head against geometric-mean token confidence on 2,500 ChartQA test questions, with the 72-fit sweep behind the winning heads |
| [02_multidataset_qhead_report.html](approach1/02_multidataset_qhead_report.html) | Extension to more datasets and the unified head | Opens with a summary of the original study and its numbers, then covers the datasets (with example images), the unified head and its loss, and per-dataset results with case studies, operating points, calibration, sweep analysis, interpretation, and limits. The main report to read |
| [03_qhead_comparison_tables.md](approach1/03_qhead_comparison_tables.md) | Same stage, as plain tables | Five scorers side by side per dataset: token confidence, separate pre-generation head, separate verifier, unified pre-generation mode, unified joint mode |
| [05_mixture_new_loss_report.html](approach1/05_mixture_new_loss_report.html) | Training across datasets | One head trained on the pooled mixture of all four datasets; sequential training (train on A, then B, then C) with forgetting measured for the old and the new loss, with and without replay; and a new loss whose extra term makes the score fall when the image is corrupted. Includes a 2 x 2 ablation of its two contrastive terms and a check of how well the answer-token probability tracks the image. Opens with a recap of everything done in Approach 1 and has a dedicated section on the new loss: its equation, what each term asks for, how it is measured, and the grounding term on its own |
| [06_corruption_new_loss_report.html](approach1/06_corruption_new_loss_report.html) | The new loss with a corrupted image | The same loss as report 05, with the corrupted state taken from the same image after noise is mixed in instead of from another example's image. Covers the pooled mixture, each dataset alone, sequential training with forgetting, and a check of whether grounding learned under swap or noise carries over to mask and shuffle corruptions that were never trained on. Ends with the limits of the result |
| [07_correction_experiments_report.html](approach1/07_correction_experiments_report.html) | Acting on the head | First experiments that use the exported unified head to change answers: prompted retry, scored retry and reranking, best-of-five candidate selection, and a deeper second pass, on 100 balanced test examples, with the oracle upper bound for each. No step-by-step reasoning is generated, so this is prompted self-review and not chain of thought |
| [08_approach1_complete_report.html](approach1/08_approach1_complete_report.html) and [PDF](approach1/08_approach1_complete_report.pdf), with [08_approach1_results.xlsx](approach1/08_approach1_results.xlsx) | The whole of Approach 1 | One document for the mid-point report and presentation: every stage in order with the reason for it, the math, head parameter counts, all results (original ChartQA study, four datasets, unified head, pooled mixture, old and new loss with swap and noise corruption, sequential forgetting), the trained heads, and the correction experiments, with limits, next steps and a slide-ready summary. Math is embedded, so it renders offline and prints to PDF. The Excel workbook holds every results table, one sheet per table (35 sheets: metrics, gate curves, sweeps, winners, grounding, sequential matrices, head manifests, correction experiments, timings), and can also be downloaded from the top of the report |
| [09_heads_vs_linear_probes.md](approach1/09_heads_vs_linear_probes.md) | Trained heads against linear probes | Per dataset, head kind and mode: the best linear probe against the best trained head, with AUROC, error AUPRC, the architecture and parameter count of the head, the difference, and whether it exceeds the seed spread. Both training settings (pooled mixture, each dataset alone). Also in the complete report (Section 8.7) and the Excel workbook |
| [04_sweep_data/](approach1/04_sweep_data/) | Machine-readable data behind 02 and 03 | Per dataset: `*_separate_heads_sweep.json`, `*_unified_head_sweep.json` (every run and the winner), and `*_analysis.json` (curves, reliability, categories, case studies, bootstrap intervals). Also `chartqa_original_validation_runs.json`, the 72 validation runs of the original study parsed from report 01, and for the mixture: `mixture_<kind>_heads_sweep.json` (every run and winner), `mixture_<kind>_sequential.json` (the sequential-training matrices and retention metrics), and `mixture_analysis.json`. For report 06: `corruption_<tag>_<kind>_sweep.json` for the noise mixture (`mixture_noise`), the swap mixture graded on the other corruptions (`mixture_xeval`), and each dataset alone with swap and with noise (`<dataset>_swap`, `<dataset>_noise`), plus `corruption_mixture_noise_<kind>_sequential.json` and `corruption_mixture_noise_analysis.json` |

Reading order: 01 for the starting point, 02 for the per-dataset results, then 05 for training across datasets. Open the HTML
reports in a browser. They are self-contained and follow the system light or dark setting.

Headline numbers (Qwen3-VL-2B, test correctness AUROC):

| Dataset | Token confidence | Best separate head | Unified head, joint mode |
|---|---:|---:|---:|
| ChartQA (original study, report 01) | 0.750 | 0.895 | not run |
| ChartQA (rerun, same protocol as the rows below) | 0.751 | 0.895 | 0.898 |
| TextVQA | 0.726 | 0.868 | 0.867 |
| DocVQA | 0.685 | 0.788 | 0.811 |
| ScienceQA | 0.654 | 0.918 | 0.925 |

The raw sweep outputs of the original study were not committed with report 01, so
`04_sweep_data/` holds its validation runs as parsed from that report. ChartQA was rerun under
the same protocol as the other datasets, and that rerun reproduces the original numbers (token
confidence 0.751 against 0.750, linear pre-generation probe 0.895 against 0.895). Its full sweep
and analysis files are in `04_sweep_data/` with the others.

## Approach 2: verifier reranking

No results yet. See [approach2/README.md](approach2/README.md).
