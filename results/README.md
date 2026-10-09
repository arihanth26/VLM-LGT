# Results

Published results for the project, numbered in the order the work happened. Working copies
stay where the code writes them (`data/approach1_latent_self_correction/results/` and
`.../q_models/`). The files here are copies, refreshed with:

```
python -m data.approach1_latent_self_correction.publish_results
```

## Approach 1: latent self-correction

| File | Stage | What it shows |
|---|---|---|
| [01_chartqa_original_study_report.html](approach1/01_chartqa_original_study_report.html) | First study, ChartQA only | A learned correctness head against geometric-mean token confidence on 2,500 ChartQA test questions, with the 72-fit sweep behind the winning heads |
| [02_multidataset_qhead_report.html](approach1/02_multidataset_qhead_report.html) | Extension to more datasets and the unified head | Opens with a summary of the original study and its numbers, then covers the datasets (with example images), the unified head and its loss, and per-dataset results with case studies, operating points, calibration, sweep analysis, interpretation, and limits. The main report to read |
| [03_qhead_comparison_tables.md](approach1/03_qhead_comparison_tables.md) | Same stage, as plain tables | Five scorers side by side per dataset: token confidence, separate pre-generation head, separate verifier, unified pre-generation mode, unified joint mode |
| [04_sweep_data/](approach1/04_sweep_data/) | Machine-readable data behind 02 and 03 | Per dataset: `*_separate_heads_sweep.json`, `*_unified_head_sweep.json` (every run and the winner), and `*_analysis.json` (curves, reliability, categories, case studies, bootstrap intervals). Also `chartqa_original_validation_runs.json`, the 72 validation runs of the original study parsed from report 01 |

Reading order: 01 for the starting point, then 02 for the current state. Open the HTML
reports in a browser. They are self-contained and follow the system light or dark setting.

Headline numbers (Qwen3-VL-2B, test correctness AUROC):

| Dataset | Token confidence | Best separate head | Unified head, joint mode |
|---|---:|---:|---:|
| ChartQA (original study, report 01) | 0.750 | 0.895 | not run yet |
| TextVQA | 0.726 | 0.868 | 0.867 |
| DocVQA | 0.685 | 0.788 | 0.811 |
| ScienceQA | 0.654 | 0.918 | 0.925 |

The raw sweep outputs of the original study were not committed with report 01, so
`04_sweep_data/` holds its validation runs as parsed from that report. A rerun of ChartQA under
the same protocol as the other datasets will add full sweep and analysis files.

## Approach 2: verifier reranking

No results yet. See [approach2/README.md](approach2/README.md).
