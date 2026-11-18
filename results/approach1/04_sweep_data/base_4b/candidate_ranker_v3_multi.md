# Candidate ranker v3: reasoning and crops in one ranker, with source and dataset features

Chosen on validation: `T1_k0_ground_cot_multi` with `ranker_v3_scalars_nods` (validation gain +1.64 points). Questions: {'train': 12000, 'val': 6420, 'test': 9517}. Baseline to beat: v2 `T1_k0_cot` scalars_judge, +1.84 on test.

| Dataset | Greedy | Chosen | Difference (95% interval) | Fixed / broken | p | Oracle |
|---|---:|---:|---|---|---:|---:|
| pooled | 85.84% | 87.55% | +1.71 (+1.37 to +2.05) | +227 / -64 | 0.0000 | 93.34% |
| chartqa_full | 81.16% | 84.40% | +3.24 (+2.40 to +4.08) | +97 / -16 | 0.0000 | 91.60% |
| textvqa | 81.64% | 82.64% | +1.00 (+0.32 to +1.68) | +47 / -22 | 0.0035 | 89.48% |
| docvqa | 89.12% | 90.28% | +1.16 (+0.52 to +1.80) | +48 / -19 | 0.0005 | 95.36% |
| scienceqa_img | 92.76% | 94.15% | +1.39 (+0.79 to +2.03) | +35 / -7 | 0.0000 | 97.77% |

## Per-dataset router (chosen on validation, untrained)

| Dataset | Configuration | Selector | Val accuracy | Test difference (95% interval) |
|---|---|---|---:|---|
| chartqa_full | `T1_k0_cot_multi` | `ranker_v3_scalars_nods` | 81.09% | +3.92 (+3.00 to +4.88) |
| textvqa | `T1_k0_zoom_ground` | `ranker_v3_scalars_ds` | 81.87% | +1.12 (+0.56 to +1.72) |
| docvqa | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 91.93% | +1.16 (+0.52 to +1.80) |
| scienceqa_img | `T1_k0_cot_multi` | `ranker_v3_scalars_ds` | 94.47% | +1.54 (+0.79 to +2.28) |
| pooled | | | | +1.95 (+1.60 to +2.31), accuracy 87.79% |

## Accuracy against cost (each row chosen on validation within the budget, read on test)

Cost is model generate calls per question (greedy, samples, correction sources; the grounded crops share one box call). Candidates is the mean number of distinct candidates, each needing one encoding pass and one judge pass (no generation).

| Budget (calls) | Configuration | Selector | Calls | Candidates | Val accuracy | Test accuracy | Test difference (95% interval) |
|---:|---|---|---:|---:|---:|---:|---|
| 2 | `T1_k0_cot` | `ranker_v3_scalars_ds` | 2 | 1.26 | 86.42% | 87.28% | +1.44 (+1.05 to +1.81) |
| 3 | `T1_k0_cot` | `ranker_v3_scalars_ds` | 2 | 1.26 | 86.42% | 87.28% | +1.44 (+1.05 to +1.81) |
| 4 | `T1_k0_cot` | `ranker_v3_scalars_ds` | 2 | 1.26 | 86.42% | 87.28% | +1.44 (+1.05 to +1.81) |
| 5 | `T1_k0_cot_multi` | `ranker_v3_scalars_ds` | 5 | 1.50 | 86.62% | 87.50% | +1.66 (+1.31 to +2.01) |
| 6 | `T1_k0_cot_multi` | `ranker_v3_scalars_ds` | 5 | 1.50 | 86.62% | 87.50% | +1.66 (+1.31 to +2.01) |
| 7 | `T1_k0_cot_multi` | `ranker_v3_scalars_ds` | 5 | 1.50 | 86.62% | 87.50% | +1.66 (+1.31 to +2.01) |
| 8 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |
| 9 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |
| 10 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |
| 11 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |
| 12 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |
| 13 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |
| 14 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |
| 16 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |
| 19 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 8 | 1.92 | 86.68% | 87.55% | +1.71 (+1.37 to +2.05) |

## Every configuration on test (pooled accuracy)

| Configuration | greedy | head_best | judge_best | ranker_v3_scalars_ds | ranker_v3_scalars_nods | oracle |
|---|---:|---:|---:|---:|---:|---:|
| T1_k0_zoom | 85.84% | 85.84% | 84.28% | 86.16% | 86.11% | 91.15% |
| T1_k0_vcd | 85.84% | 85.84% | 85.81% | 85.96% | 85.94% | 86.75% |
| T1_k0_first_round | 85.84% | 85.84% | 84.23% | 86.20% | 86.22% | 91.49% |
| T1_k0_ground | 85.84% | 85.84% | 85.68% | 86.42% | 86.47% | 89.85% |
| T1_k0_cot | 85.84% | 85.84% | 86.34% | 87.28% | 87.40% | 90.25% |
| T1_k0_ground_cot | 85.84% | 85.84% | 85.70% | 87.28% | 87.28% | 92.41% |
| T1_k0_zoom_ground | 85.84% | 85.84% | 84.22% | 86.51% | 86.46% | 92.64% |
| T1_k0_everything | 85.84% | 85.84% | 84.14% | 87.35% | 87.48% | 94.49% |
| T1_k0_zoom_cot | 85.84% | 85.84% | 84.41% | 87.28% | 87.15% | 93.64% |
| T1_k0_zoom_ground_cot | 85.84% | 85.84% | 84.18% | 87.24% | 87.39% | 94.33% |
| T1_k0_cot_multi | 85.84% | 85.84% | 85.89% | 87.50% | 87.71% | 91.85% |
| T1_k0_ground_cot_multi | 85.84% | 85.84% | 85.30% | 87.56% | 87.55% | 93.34% |
| T1_k0_zoom_cot_multi | 85.84% | 85.84% | 84.06% | 87.48% | 87.44% | 94.38% |
| T1_k0_everything_multi | 85.84% | 85.84% | 83.85% | 87.67% | 87.74% | 95.05% |
| T1_k5_none | 85.84% | 85.84% | 86.07% | 86.18% | 86.13% | 88.37% |
| T1_k5_zoom | 85.84% | 85.84% | 84.44% | 86.34% | 86.26% | 92.23% |
| T1_k5_vcd | 85.84% | 85.84% | 86.04% | 86.16% | 86.19% | 88.63% |
| T1_k5_first_round | 85.84% | 85.84% | 84.38% | 86.26% | 86.29% | 92.34% |
| T1_k5_ground | 85.84% | 85.84% | 85.57% | 86.52% | 86.48% | 90.97% |
| T1_k5_cot | 85.84% | 85.84% | 86.19% | 87.23% | 87.36% | 91.58% |
| T1_k5_ground_cot | 85.84% | 85.84% | 85.53% | 87.14% | 87.37% | 93.05% |
| T1_k5_zoom_ground | 85.84% | 85.84% | 84.20% | 86.61% | 86.53% | 93.26% |
| T1_k5_everything | 85.84% | 85.84% | 84.11% | 87.31% | 87.50% | 94.74% |
| T1_k5_zoom_cot | 85.84% | 85.84% | 84.39% | 87.07% | 87.03% | 94.17% |
| T1_k5_zoom_ground_cot | 85.84% | 85.84% | 84.12% | 87.25% | 87.30% | 94.66% |
| T1_k5_cot_multi | 85.84% | 85.84% | 85.66% | 87.31% | 87.40% | 92.75% |
| T1_k5_ground_cot_multi | 85.84% | 85.84% | 85.11% | 87.32% | 87.46% | 93.87% |
| T1_k5_zoom_cot_multi | 85.84% | 85.84% | 84.01% | 87.21% | 87.20% | 94.78% |
| T1_k5_everything_multi | 85.84% | 85.84% | 83.79% | 87.44% | 87.67% | 95.24% |
